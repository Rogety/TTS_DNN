# 從 lab 檔中 抽出 duration, acoustic states, basic acoustic states

require 'json'
require 'optparse'
require_relative 'question'


# options
options = {}
options[:gen] = false
options[:source] = false
options[:target] = false

OptionParser.new do |opts|
    opts.on("-g", "--gen", "gen dab in the gen dir") do |v|
        options[:gen] = true
    end
    opts.on("-s", "--source", "gen dab in the source dir") do |v|
        options[:source] = true
    end
    opts.on("-t", "--target", "gen dab in the target dir") do |v|
        options[:target] = true
    end
end.parse!

# configs
$configs = JSON.load(File.read( "configs/Configs.json" ))
$configs_dir_path = "configs"
$question_dir_path = "data/questions"
$number_of_states = $configs["number_of_states"]
$frame_shift = $configs["frame_shift"]*(10**7).to_f
$conf_filename = File.join($question_dir_path, $configs["question_name"])
$speed = $configs["speed"]

# 建立 question set
$questions = Array.new()
$reserved_questions = Array.new()
qconfig = File.open($conf_filename,"r")
qconfig.each_line do |line|
    if line =~ /^\s*#/ then
        next
    end
    question = Question.new(line)
    if question.is_reserved? then
        $reserved_questions.push(question)
    else
        $questions.push(question)
    end
end
qconfig.close

$q_state_in_phone_fwd = nil
$q_state_in_phone_bwd = nil
$q_frame_in_state_fwd = nil
$q_frame_in_state_bwd = nil
$q_frame_in_phone_fwd = nil
$q_frame_in_phone_bwd = nil

$reserved_questions.each do |q|
    if q.name == "Pos_C-State_in_Phone(Fw)" then
        $q_state_in_phone_fwd = q
    elsif q.name == "Pos_C-State_in_Phone(Bw)" then
        $q_state_in_phone_bwd = q
    elsif q.name == "Pos_C-Frame_in_State(Fw)" then
        $q_frame_in_state_fwd = q
    elsif q.name == "Pos_C-Frame_in_State(Bw)" then
        $q_frame_in_state_bwd = q
    elsif q.name == "Pos_C-Frame_in_Phone(Fw)" then
        $q_frame_in_phone_fwd = q
    elsif q.name == "Pos_C-Frame_in_Phone(Bw)" then
        $q_frame_in_phone_bwd = q
    end
end

def gen_dab( base_dir_path )
    lab_state_dir = File.join(base_dir_path, "lab", "state")
    a_dir_path = File.join(base_dir_path, "set", "a")
    b_dir_path = File.join(base_dir_path, "set", "b")
    d_dir_path = File.join(base_dir_path, "set", "d")

    Dir.glob(File.join(lab_state_dir,"*.lab")).sort.each do |f|
        base = File.basename(f, ".lab")
        phones = File.readlines(f).each_slice($number_of_states).to_a

        if phones.last.size != $number_of_states then
            puts "missing state in #{base}"
            next
        end

        # 0~max 共 max+1種的frame數
        max_frame_in_state = $q_frame_in_state_fwd.max + 1

        # 語速資訊 sec. / #syl.
        syllable_length = []

        a = [] # size:[number_of_frames x a_order]
        b = [] # size:[number_of_states x b_order]
        d = [] # size:[number_of_states x 1]

        syllable_t0 = 0.0
        syllable_t1 = 0.0

        phones.each_with_index do |phone, phone_index|
            ans_of_cur_phone = []

            phone_t0 = phone.first.split[0].to_f
            phone_t1 = phone.last.split[1].to_f
            phone_str = phone.first.split[2]
            total_frames_in_phone = (phone_t1-phone_t0)/$frame_shift
            cur_frame_in_phone = 0

            $questions.each do |q|
                ans = q.ask(phone_str)
                ans_of_cur_phone.push(ans)
            end

            # 下面是 unportable 的方法，應該用其他方法提供語速資訊而不是用下面的方法切音節再算語速。語速應該放在 question set 中 。

            ## 確認現在是不是音節開頭

            ## 現在不是 sil 或是 sp
            if phone_str.match?(/.*-(sil|sp)\+.*/) == false then
                ## 現在不是句子開頭
                if phone_index > 0 then
                    if phone_str.match?(/\A(sil|sp)-.*/) then
                        syllable_t0 = phone_t0
                    else
                        ## 檢查前一個和現在的 p q r pb nb tone 是否相同
                        prev_phone_str = phones[phone_index-1].first.split[2]
                        if phone_str[/\/p:.*\Z/] != prev_phone_str[/\/p:.*\Z/] then
                            ## 是音節開頭
                            syllable_t0 = phone_t0
                        end
                    end
                else
                    ## 是音節開頭
                    syllable_t0 = phone_t0
                end
            end

            ## 確認現在是不是音節結尾

            ## 現在不是 sil 或是 sp
            if phone_str.match?(/.*-(sil|sp)\+.*/) == false then
                ## 現在不是句子結尾
                if phone_index < phones.length - 1 then
                    ## 檢查前一個和現在的 p q r pb nb tone 是否相同
                    if phone_str.match?(/.*\+(sil|sp)\/p:.*/) then
                        syllable_t1 = phone_t1
                        syllable_length.push( (syllable_t1-syllable_t0)/(10**7) )
                    else
                        next_phone_str = phones[phone_index+1].first.split[2]
                        if phone_str[/\/p:.*\Z/] != next_phone_str[/\/p:.*\Z/] then
                            ## 是音節結尾
                            syllable_t1 = phone_t1
                            syllable_length.push( (syllable_t1-syllable_t0)/(10**7) )
                        end
                    end
                else
                    ## 是音節結尾
                    syllable_t1 = phone_t1
                    syllable_length.push( (syllable_t1-syllable_t0)/(10**7) )
                end
            end

            phone.each_with_index do |state, state_index|
                ans_of_cur_state = ans_of_cur_phone.dup

                state_fwd_n = $q_state_in_phone_fwd.min + state_index
                state_bwd_n = $q_state_in_phone_bwd.max - state_index
                ans_of_cur_state.push( $q_state_in_phone_fwd.ask(state_fwd_n) )
                ans_of_cur_state.push( $q_state_in_phone_bwd.ask(state_bwd_n) )

                _split = state.split
                state_t0 = _split[0].to_f
                state_t1 = _split[1].to_f
                total_frames_in_state = (state_t1-state_t0)/$frame_shift

                (0...total_frames_in_state).each do |frame_index|
                    ans_of_cur_frame = ans_of_cur_state.dup
                    frame_in_state_fwd_n = $q_frame_in_state_fwd.min + frame_index
                    frame_in_state_bwd_n = total_frames_in_state - frame_index
                    frame_in_phone_fwd_n = $q_frame_in_phone_fwd.min + cur_frame_in_phone
                    frame_in_phone_bwd_n = total_frames_in_phone - cur_frame_in_phone

                    ans_of_cur_frame.push( $q_frame_in_state_fwd.ask(frame_in_state_fwd_n) )
                    ans_of_cur_frame.push( $q_frame_in_state_bwd.ask(frame_in_state_bwd_n) )
                    ans_of_cur_frame.push( $q_frame_in_phone_fwd.ask(frame_in_phone_fwd_n) )
                    ans_of_cur_frame.push( $q_frame_in_phone_bwd.ask(frame_in_phone_bwd_n) )

                    a.push(ans_of_cur_frame)
                    cur_frame_in_phone+=1
                end

                #ans_of_cur_state.push(speed_index)
                b.push(ans_of_cur_state)

                # 把太長的 frame 數 clip 到最大值
                if total_frames_in_state >= max_frame_in_state
                    total_frames_in_state = max_frame_in_state
                end
                d.push(total_frames_in_state)
            end
        end

        # 計算一個音節平均的秒數 並放入 b 中
        #speed = syllable_length.reduce(:+) / syllable_length.length
        speed = 0.0
        b.each do | ans_of_cur_state | ans_of_cur_state.push(speed) end

        puts("#{base} a:#{a.size}x#{a[0].size}, b:#{b.size}x#{b[0].size}, d:#{d.size}, speed:#{speed}")
        a.flatten!
        b.flatten!
        d.flatten!
        File.binwrite(File.join(a_dir_path, "#{base}.bin"), a.pack("f*"))
        File.binwrite(File.join(b_dir_path, "#{base}.bin"), b.pack("f*"))
        File.binwrite(File.join(d_dir_path, "#{base}.bin"), d.pack("l*"))
    end
end


if options[:source] then
    base_dir_path = $configs["source_dir_path"]
    gen_dab(base_dir_path)
end

if options[:target] then
    base_dir_path = $configs["target_dir_path"]
    gen_dab(base_dir_path)
end

if options[:gen] then
    base_dir_path = $configs["gen_dir_path"]
    gen_dab(base_dir_path)
end

io_config_path = File.join($configs_dir_path,"IOConfigs.json")
io_config = if File.exist?(io_config_path) then
                JSON.parse(File.read(io_config_path))
            else
                {}
            end

io_config["a_order"] = $questions.size + $reserved_questions.size
io_config["b_order"] = $questions.size + 2 + 1
io_config["d_order"] = 1
io_config["d_class"] = ($q_frame_in_state_fwd.max + 1).to_i#($q_frame_in_state_fwd.max + 1).to_i**2

File.write(File.join($configs_dir_path,"IOConfigs.json"),JSON.pretty_generate(io_config))
