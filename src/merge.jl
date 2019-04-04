using HDF5, JSON, ArgParse, Printf

function main(ARGS)
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--name"; arg_type=String; default="spatial"; help="spatial or objects")
        ("--chunksNum"; arg_type=Int; default=11; help="number of file chunks")
        ("--chunksSize"; arg_type=Int; default=10000; help="number of file chunks")
    end
    args  = parse_args(ARGS,s;as_symbols=true)
    return args
end

args = main(ARGS)
println("Merging features file for gqa_$(args[:name]) This may take a while.")

# Format specification for features files
spec = Dict("spatial" => Dict("features"=> (7, 7, 2048, 108079)),
	    "objects" => Dict("features"=> (2048, 100, 108077),
			      "bboxes"=> (4, 100, 108077))
            )


rng2ind(rng,::NTuple{N, Int}) where N = ntuple(i-> i==N ? rng : Colon(), N)

# Merge hdf5 files
h5open("./gqa_$(args[:name]).h5","w") do out
    datasets  = Dict()
    for (dname,value) in spec[args[:name]]
	datasets[dname] = d_create(out, dname, datatype(Float32), dataspace(value), "chunk", (value[1:end-1]...,1))# Array{Float32}(undef,value...)
    end

    for i in 1:args[:chunksNum]
	h5open("./$(args[:name])/gqa_$(args[:name])_$(i-1).h5","r") do chunk
	    for (dname,value) in spec[args[:name]]
	        low = (i-1)*args[:chunksSize]+1
	        high = i < args[:chunksNum] ? i*args[:chunksSize] : last(value)
	        datasets[dname][rng2ind(low:high,value)...] = read(chunk[dname])
            end
        end
    end
end

# Update info file

info = JSON.parsefile("./$(args[:name])/gqa_$(args[:name])_info.json")
for (imageId,val) in info
    info[imageId]["index"] = val["file"] * args[:chunksSize] + val["idx"] + 1
    delete!(info[imageId],"file")
end

open("./$(args[:name])/gqa_$(args[:name])_merged_info.json", "w") do infoOut
    write(infoOut,json(info))
end


    
