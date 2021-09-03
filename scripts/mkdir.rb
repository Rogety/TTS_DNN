require 'json'
require 'fileutils'

configs = JSON.load(File.read( "configs/Configs.json" ) )

source_dir_path = configs["source_dir_path"]
gen_dir_path = configs["gen_dir_path"]



[
    File.join(source_dir_path,"stat"),
    File.join(source_dir_path,"set", "a"),
    File.join(source_dir_path,"set", "b"),
    File.join(source_dir_path,"set", "c"),
    File.join(source_dir_path,"set", "d"),
    File.join(gen_dir_path,"set", "a"),
    File.join(gen_dir_path,"set", "b"),
    File.join(gen_dir_path,"set", "d"),
    File.join("model", "source"),
    File.join("gen","source","from_a"),
    File.join("gen","source","from_b"),
].each do |p|
    FileUtils.mkdir_p(p)
end
