# 從資料庫產生DNN訓練用的輸出
# 一筆資料中依序為 lf0[i], mgc[i], lf0_delta[i], mgc_delta[i], lf0_delta2[i], mgc_delta2[i], voiced[i]
# 其中 i 為時間的index

require 'json'
require 'optparse'

# options
options = {}
options[:source] = false
options[:target] = false

OptionParser.new do |opts|
    opts.on("-s", "--source", "gen c in the source dir") do |v|
        options[:source] = true
    end
    opts.on("-t", "--target", "gen c in the target dir") do |v|
        options[:target] = true
    end
end.parse!


# configs
$configs = JSON.load(File.read("configs/Configs.json"))
$configs_dir_path = "configs"
$mgc_order = $configs["mgc_order"]
$lf0_order = $configs["lf0_order"]

def gen_c( gen_dir_path )
    lf0_dir_path = File.join(gen_dir_path,"lf0")
    mgc_dir_path = File.join(gen_dir_path,"mgc")
    c_dir_path = File.join(gen_dir_path,"set","c")
    Dir.glob(File.join(lf0_dir_path, "*.lf0")).sort.each do |f|
        base = File.basename(f, ".lf0")
        puts(base)
        # 讀取資料
        lf0 = File.binread(File.join(lf0_dir_path,"#{base}.lf0" )).unpack("f*")
        mgc = File.binread(File.join(mgc_dir_path,"#{base}.mgc" )).unpack("f*").each_slice($mgc_order).to_a
        # 確認資料的長度
        if lf0.length != mgc.length then
            puts("Error: batch size inconsistant: lf0(#{lf0.length}) mgc(#{mgc.length})")
            next
        end
        if mgc.last.length != $mgc_order then
            puts("Error: the size should be the multiple of the lf0 order #{$mgc_order}")
            next
        end
        # 找出 清音/濁音
        voiced = lf0.map do |x|
            if x!=-(10**10) then 1 else 0 end
        end
        # 內插
        first, idx = lf0.each_with_index do |x, i|
            break [x, i] if x != -(10**10)
        end
        (0...idx).each do |i|
            lf0[i] = first
        end
        ## 結尾
        last, idx = lf0.to_enum.with_index.reverse_each do |x, i|
            break [x, i] if x != -(10**10)
        end
        (idx...lf0.length).each do |i|
            lf0[i] = last
        end
        ## 中間
        lf0.each_with_index do |x, i|
            if x == -(10**10) then
                count = 0 # 不包含開頭和結尾
                a = lf0[i-1]
                b = (i..lf0.length).each do |idx|
                    break lf0[idx] if lf0[idx]!=-(10**10)
                    count+=1
                end
                (1..count).each do |c|
                    lf0[i-1+c] = a+(b-a)/count*c
                end
            end
        end
        # 計算 delta
        lf0_delta = lf0.map.with_index do |x,i|
            if i==0 then
                0.5 * lf0[i+1]
            elsif i== (lf0.size() -1) then
                -0.5 * lf0[i-1]
            else
                0.5*lf0[i+1] - 0.5*lf0[i-1]
            end
        end
        '''
        ## test mgc[0] = 0
        mgc_test = mgc.map.with_index do |x,i|
            if i==0 then
                x.map.with_index do |u,j| 0.0 end
            else
                x.map.with_index do |u,j| mgc[i][j] end
            end
        end
        '''
        mgc_delta = mgc.map.with_index do |x,i|
            if i==0 then
                x.map.with_index do |u,j| 0.5* mgc[i+1][j] end
            elsif i== (mgc.size-1) then
                x.map.with_index do |u,j| -0.5* mgc[i-1][j] end
            else
                x.map.with_index do |u,j| 0.5*mgc[i+1][j]-0.5*mgc[i-1][j] end
            end
        end
        # 計算 delta2
        lf0_delta2 = lf0.map.with_index do |x,i|
            if i==0 then
                -2.0*x + lf0[i+1]
            elsif i==(lf0.size-1) then
                -2.0*x + lf0[i-1]
            else
                lf0[i-1] -2.0*x + lf0[i+1]
            end
        end
        mgc_delta2 = mgc.map.with_index do |x,i|
            if i==0 then
                x.map.with_index do |u,j| -2.0*u+mgc[i+1][j] end
            elsif i==(mgc.size-1) then
                x.map.with_index do |u,j| -2.0*u+mgc[i-1][j] end
            else
                x.map.with_index do |u,j| mgc[i-1][j]-2.0*u+mgc[i+1][j] end
            end
        end
        # make cmp
        cmp =[]
        for i in 0...lf0.length
            item = [lf0[i], mgc[i], lf0_delta[i], mgc_delta[i], lf0_delta2[i], mgc_delta2[i], voiced[i] ].flatten!
            #item = [lf0[i], mgc_test[i], lf0_delta[i], mgc_delta[i], lf0_delta2[i], mgc_delta2[i], voiced[i] ].flatten!
            cmp.push(item)
        end
        File.binwrite( File.join(c_dir_path,"#{base}.bin"), cmp.flatten.pack("f*") )
    end
end


if options[:source] then
    gen_dir_path = $configs["source_dir_path"]
    gen_c(gen_dir_path)
end


if options[:target] then
    gen_dir_path = $configs["target_dir_path"]
    gen_c(gen_dir_path)
end

io_config_path = File.join($configs_dir_path,"IOConfigs.json")
io_config = if File.exist?(io_config_path) then
                JSON.parse(File.read(io_config_path))
            else
                {}
            end

io_config["c_order"] = $mgc_order*3+$lf0_order*3+1
File.write(File.join($configs_dir_path,"IOConfigs.json"),JSON.pretty_generate(io_config))
