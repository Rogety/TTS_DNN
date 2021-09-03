require 'json'
require 'optparse'

# options
options = {}
options[:source] = false
options[:target] = false
options[:from_a] = false
options[:from_b] = false

OptionParser.new do |opts|
    opts.on("-s", "--source", "gen c in the source dir") do |v|
        options[:source] = true
    end
    opts.on("-t", "--target", "gen c in the target dir") do |v|
        options[:target] = true
    end
    opts.on("-a", "--from_a", "gen c in the from_a dir") do |v|
        options[:from_a] = true
    end
    opts.on("-b", "--from_b", "gen c in the from_b dir") do |v|
        options[:from_b] = true
    end
end.parse!

configs = JSON.load(File.read( "configs/Configs.json" ) )

sr = configs["sampling_rate"]
fs = (configs["frame_shift"]*sr).to_i
gm = configs["gamma"]
fw = configs["frequency_warping"]
fft = configs["fft_len"]
mgc_order = configs["mgc_order"]

if options[:source] then 
    gen_dir_path = if options[:from_a] then
        File.join("gen","source","from_a")
    elsif options[:from_b] then
        File.join("gen","source","from_b")
    else 
        ""
    end

    Dir.glob(File.join(gen_dir_path,"*.uv")).sort.each do |f|
        base = File.basename(f,".uv")
        puts(base)
        # output
        lf0_fn = File.join(gen_dir_path, "#{base}.lf0")
        f0_fn = File.join(gen_dir_path, "#{base}.f0")
        mgc_fn = File.join(gen_dir_path, "#{base}.mgc")

        # input
        uv_fn = File.join(gen_dir_path, "#{base}.uv")
        f0_mlpg_fn = File.join(gen_dir_path, "#{base}.lf0.mlpg")
        mgc_mlpg_fn = File.join(gen_dir_path, "#{base}.mgc.mlpg")

        # apply mlpg to f0
        line = `mlpg -m 0 -d -0.5 0 0.5 -d 1 -2 1 0 0 #{f0_mlpg_fn} | sopr -EXP -INV -m #{sr} | vopr -m #{uv_fn} > #{f0_fn}`
        
        # make lf0 for check
        line = `mlpg -m 0 -d -0.5 0 0.5 -d 1 -2 1 0 0 #{f0_mlpg_fn} | sopr -EXP -INV -m #{sr} | vopr -m #{uv_fn} > #{lf0_fn}`

        # apply mlpg to mgc
        line = `mlpg -m #{mgc_order-1} -d -0.5 0 0.5 -d 1 -2 1 0 0 #{mgc_mlpg_fn} > #{mgc_fn}`
    
        # synthesis
        line = `excite -n -p #{fs} #{f0_fn} | mglsadf -P 5 -m #{mgc_order-1} -p #{fs} -a #{fw} -c #{gm} #{mgc_fn}| x2x +fs -o > #{File.join(gen_dir_path, "#{base}.raw")}`
        line = `raw2wav -s #{(sr/1000).to_i} -d #{gen_dir_path} #{File.join(gen_dir_path, "#{base}.raw")}`
    end
end

if options[:target] then 
    gen_dir_path = if options[:from_a] then
        File.join("gen","target","from_a")
    elsif options[:from_b] then
        File.join("gen","target","from_b")
    else 
        ""
    end

    Dir.glob(File.join(gen_dir_path,"*.uv")).sort.each do |f|
        base = File.basename(f,".uv")
        puts(base)
        # output
        lf0_fn = File.join(gen_dir_path, "#{base}.lf0")
        f0_fn = File.join(gen_dir_path, "#{base}.f0")
        mgc_fn = File.join(gen_dir_path, "#{base}.mgc")

        # input
        uv_fn = File.join(gen_dir_path, "#{base}.uv")
        f0_mlpg_fn = File.join(gen_dir_path, "#{base}.lf0.mlpg")
        mgc_mlpg_fn = File.join(gen_dir_path, "#{base}.mgc.mlpg")

        # apply mlpg to f0
        line = `mlpg -m 0 -d -0.5 0 0.5 -d 1 -2 1 0 0 #{f0_mlpg_fn} | sopr -EXP -INV -m #{sr} | vopr -m #{uv_fn} > #{f0_fn}`
        
        # make lf0 for check
        line = `mlpg -m 0 -d -0.5 0 0.5 -d 1 -2 1 0 0 #{f0_mlpg_fn} | sopr -EXP -INV -m #{sr} | vopr -m #{uv_fn} > #{lf0_fn}`

        # apply mlpg to mgc
        line = `mlpg -m #{mgc_order-1} -d -0.5 0 0.5 -d 1 -2 1 0 0 #{mgc_mlpg_fn} > #{mgc_fn}`
    
        # synthesis
        line = `excite -n -p #{fs} #{f0_fn} | mglsadf -P 5 -m #{mgc_order-1} -p #{fs} -a #{fw} -c #{gm} #{mgc_fn}| x2x +fs -o > #{File.join(gen_dir_path, "#{base}.raw")}`
        line = `raw2wav -s #{(sr/1000).to_i} -d #{gen_dir_path} #{File.join(gen_dir_path, "#{base}.raw")}`
    end
end