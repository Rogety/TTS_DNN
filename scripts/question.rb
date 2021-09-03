def patch_str(str)
  '^'+str
    .gsub(/(\\)/,'\\\\')
    .gsub(/(\.)/,'\\\\.')
    .gsub(/(\^)/,'\\\\^')
    .gsub(/(\$)/,'\\\\$')
    .gsub(/(\+)/,'\\\\+')
    .gsub(/(\?)/,'\\\\?')
    .gsub(/(\()/,'\\\\(')
    .gsub(/(\))/,'\\\\)')
    .gsub(/(\{)/,'\\\\{')
    .gsub(/(\})/,'\\\\}')
    .gsub(/(\[)/,'\\\\[')
    .gsub(/(\])/,'\\\\]')
    .gsub(/(\|)/,'\\\\|')
    .gsub(/(\=)/,'\\\\=')
    .gsub(/(\#)/,'\\\\#')
    .gsub(/(\*)/,'.*') +'.*$'  
end

class Question
    def initialize( cfg_line )
      args = cfg_line.split
      @name = args[0]
      
      if args.size == 2 then
        @type = 'binary'
        @patts = args[1].gsub(/^{(.*)}$/,'\1').split(",") 
        @patts.map! do |str|
          patch_str(str)
        end
        @patts.map! do | str |
          Regexp.new( str )
        end
      elsif args.size == 3 then
        @type = 'reserved'

        raise "can't find MIN" if args[1].scan(/MIN=/).size != 1 
        raise "can't find MAX" if args[2].scan(/MAX=/).size != 1

        @min = args[1].gsub(/MIN=([0-9]+)/,'\1').to_f
        @max = args[2].gsub(/MAX=([0-9]+)/,'\1').to_f

      elsif args.size == 4 then   
        @type = 'numerical'
        raise "can't find MIN" if args[2].scan(/MIN=/).size != 1 
        raise "can't find MAX" if args[3].scan(/MAX=/).size != 1
        @min = args[2].gsub(/MIN=([0-9]+)/,'\1').to_f
        @max = args[3].gsub(/MAX=([0-9]+)/,'\1').to_f
        
        @patts = args[1].gsub(/^{(.*)}$/,'\1').split(",") 
        @patts.map! do |str|
          patch_str(str)
        end

        raise "The numerical question cannot have multiple patterns" if @patts.size != 1

        @patts.map! do | str |
          raise "no %d or multiple %d" if str.scan(/%d/).size != 1
          str.gsub(/%d/,'([0-9]+)')
        end
        @patts.map! do | str |
          Regexp.new( str )
        end
      else 
        raise "Invalid format"
      end     
    end

    def is_reserved?()
      @type == 'reserved'
    end

    def name
      @name
    end
    def max 
      if @type == 'binary' then 
        1
      else
        @max
      end
    end
    def min
      if @type == 'binary' then
        0
      else 
        @min
      end
    end
    def ask( line )
      if @type == 'binary' then
        matched = 0.0
        @patts.map do |patt|
          if patt.match(line) != nil then
            matched = 1.0
            break
          end 
        end
        matched
      elsif @type == 'reserved' then
        (line.to_f-@min )/(@max-@min)
      elsif @type == 'numerical' then
        value = 0.0
        @patts.map do |patt|
          v = patt.match(line)
          if v == nil then 
            raise "can't match question #{patt}"
          end

          if v[1].to_f<@min || v[1].to_f>@max then
            puts("warning: #{@name} value out of range")
          end

          value = ( v[1].to_f-@min )/(@max-@min)
        end
        value
      else
        raise "invalid quesiton"
      end
    end

    def marshal_dump
      [@name, @type, @patts, @min, @max]
    end

    def marshal_load(array)
      @name, @type, @patts, @min, @max = array
    end
end