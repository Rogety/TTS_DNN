require 'json'
require 'optparse'

# options
options = {}
options[:source] = false
options[:target] = false

OptionParser.new do |opts|
    opts.on("-s", "--source", "gen trn val tst sets in the source dir") do |v|
        options[:source] = true
    end
    opts.on("-t", "--target", "gen trn val tst sets in the target dir") do |v|
        options[:target] = true
    end
end.parse!

# read configs
$io_configs = JSON.load(File.read("configs/IOConfigs.json"))

if options[:source] then 
    set_dir_path = File.join("data","source","set")
    a_trn_list = []
    c_trn_list = []
    b_trn_list = []
    d_trn_list = []

    a_val_list = []
    c_val_list = []
    b_val_list = []
    d_val_list = []

    a_tst_list = []
    c_tst_list = []
    b_tst_list = []
    d_tst_list = []

    Dir.glob(File.join(set_dir_path, "a", "*")).sort.each do |fn|
        base = File.basename(fn)
        if base.end_with?("7.bin","8.bin") then 
            a_val_list.push(File.join(set_dir_path, "a", base))
            b_val_list.push(File.join(set_dir_path, "b", base))
            c_val_list.push(File.join(set_dir_path, "c", base))
            d_val_list.push(File.join(set_dir_path, "d", base))
        elsif base.end_with?("9.bin","0.bin") then
            a_tst_list.push(File.join(set_dir_path, "a", base))
            b_tst_list.push(File.join(set_dir_path, "b", base))
            c_tst_list.push(File.join(set_dir_path, "c", base))
            d_tst_list.push(File.join(set_dir_path, "d", base))
        else
            a_trn_list.push(File.join(set_dir_path, "a", base))
            b_trn_list.push(File.join(set_dir_path, "b", base))
            c_trn_list.push(File.join(set_dir_path, "c", base))
            d_trn_list.push(File.join(set_dir_path, "d", base))
        end
    end
    File.open(File.join(set_dir_path, "a_val.bin"), mode="wb") do |set|
        a_val_list.each do |file|
            puts "cat #{file} to a_val.bin"
            set.write( File.binread(file) )
        end
    end
    File.open(File.join(set_dir_path, "a_tst.bin"), mode="wb") do |set|
        a_tst_list.each do |file|
            puts "cat #{file} to a_tst.bin"
            set.write( File.binread(file) )
        end
    end
    File.open(File.join(set_dir_path, "a_trn.bin"), mode="wb") do |set|
        a_trn_list.each do |file|
            puts "cat #{file} to a_trn.bin"
            set.write( File.binread(file) )
        end
    end
    
    File.open(File.join(set_dir_path, "b_val.bin"), mode="wb") do |set|
        b_val_list.each do |file|
            puts "cat #{file} to b_val.bin"
            set.write( File.binread(file) )
        end
    end
    File.open(File.join(set_dir_path, "b_tst.bin"), mode="wb") do |set|
        b_tst_list.each do |file|
            puts "cat #{file} to b_tst.bin"
            set.write( File.binread(file) )
        end
    end
    File.open(File.join(set_dir_path, "b_trn.bin"), mode="wb") do |set|
        b_trn_list.each do |file|
            puts "cat #{file} to b_trn.bin"
            set.write( File.binread(file) )
        end
    end

    File.open(File.join(set_dir_path, "c_val.bin"), mode="wb") do |set|
        c_val_list.each do |file|
            puts "cat #{file} to c_val.bin"
            set.write( File.binread(file) )
        end
    end
    File.open(File.join(set_dir_path, "c_tst.bin"), mode="wb") do |set|
        c_tst_list.each do |file|
            puts "cat #{file} to c_tst.bin"
            set.write( File.binread(file) )
        end
    end
    File.open(File.join(set_dir_path, "c_trn.bin"), mode="wb") do |set|
        c_trn_list.each do |file|
            puts "cat #{file} to c_trn.bin"
            set.write( File.binread(file) )
        end
    end

    File.open(File.join(set_dir_path, "d_val.bin"), mode="wb") do |set|
        d_val_list.each do |file|
            puts "cat #{file} to d_val.bin"
            set.write( File.binread(file) )
        end
    end
    File.open(File.join(set_dir_path, "d_tst.bin"), mode="wb") do |set|
        d_tst_list.each do |file|
            puts "cat #{file} to d_tst.bin"
            set.write( File.binread(file) )
        end
    end
    File.open(File.join(set_dir_path, "d_trn.bin"), mode="wb") do |set|
        d_trn_list.each do |file|
            puts "cat #{file} to d_trn.bin"
            set.write( File.binread(file) )
        end
    end
end

if options[:target] then
    set_dir_path = File.join("data","target","set")
    a_trn_list = []
    c_trn_list = []
    b_trn_list = []
    d_trn_list = []

    a_val_list = []
    c_val_list = []
    b_val_list = []
    d_val_list = []

    a_tst_list = []
    c_tst_list = []
    b_tst_list = []
    d_tst_list = []

    Dir.glob(File.join(set_dir_path, "a", "*")).sort.each do |fn|
        base = File.basename(fn)
        if base.end_with?("7.bin","8.bin") then 
            a_val_list.push(File.join(set_dir_path, "a", base))
            b_val_list.push(File.join(set_dir_path, "b", base))
            c_val_list.push(File.join(set_dir_path, "c", base))
            d_val_list.push(File.join(set_dir_path, "d", base))
        elsif base.end_with?("9.bin","0.bin") then
            a_tst_list.push(File.join(set_dir_path, "a", base))
            b_tst_list.push(File.join(set_dir_path, "b", base))
            c_tst_list.push(File.join(set_dir_path, "c", base))
            d_tst_list.push(File.join(set_dir_path, "d", base))
        else
            a_trn_list.push(File.join(set_dir_path, "a", base))
            b_trn_list.push(File.join(set_dir_path, "b", base))
            c_trn_list.push(File.join(set_dir_path, "c", base))
            d_trn_list.push(File.join(set_dir_path, "d", base))
        end
    end

    File.open(File.join(set_dir_path, "a_val.bin"), mode="wb") do |set|
        a_val_list.each do |file|
            puts "cat #{file} to a_val.bin"
            set.write( File.binread(file) )
        end
    end
    File.open(File.join(set_dir_path, "a_tst.bin"), mode="wb") do |set|
        a_tst_list.each do |file|
            puts "cat #{file} to a_tst.bin"
            set.write( File.binread(file) )
        end
    end
    File.open(File.join(set_dir_path, "a_trn.bin"), mode="wb") do |set|
        a_trn_list.each do |file|
            puts "cat #{file} to a_trn.bin"
            set.write( File.binread(file) )
        end
    end
    
    File.open(File.join(set_dir_path, "b_val.bin"), mode="wb") do |set|
        b_val_list.each do |file|
            puts "cat #{file} to b_val.bin"
            set.write( File.binread(file) )
        end
    end
    File.open(File.join(set_dir_path, "b_tst.bin"), mode="wb") do |set|
        b_tst_list.each do |file|
            puts "cat #{file} to b_tst.bin"
            set.write( File.binread(file) )
        end
    end
    File.open(File.join(set_dir_path, "b_trn.bin"), mode="wb") do |set|
        b_trn_list.each do |file|
            puts "cat #{file} to b_trn.bin"
            set.write( File.binread(file) )
        end
    end

    File.open(File.join(set_dir_path, "c_val.bin"), mode="wb") do |set|
        c_val_list.each do |file|
            puts "cat #{file} to c_val.bin"
            set.write( File.binread(file) )
        end
    end
    File.open(File.join(set_dir_path, "c_tst.bin"), mode="wb") do |set|
        c_tst_list.each do |file|
            puts "cat #{file} to c_tst.bin"
            set.write( File.binread(file) )
        end
    end
    File.open(File.join(set_dir_path, "c_trn.bin"), mode="wb") do |set|
        c_trn_list.each do |file|
            puts "cat #{file} to c_trn.bin"
            set.write( File.binread(file) )
        end
    end

    File.open(File.join(set_dir_path, "d_val.bin"), mode="wb") do |set|
        d_val_list.each do |file|
            puts "cat #{file} to d_val.bin"
            set.write( File.binread(file) )
        end
    end
    File.open(File.join(set_dir_path, "d_tst.bin"), mode="wb") do |set|
        d_tst_list.each do |file|
            puts "cat #{file} to d_tst.bin"
            set.write( File.binread(file) )
        end
    end
    File.open(File.join(set_dir_path, "d_trn.bin"), mode="wb") do |set|
        d_trn_list.each do |file|
            puts "cat #{file} to d_trn.bin"
            set.write( File.binread(file) )
        end
    end
end