import Pkg; Pkg.activate("."); Pkg.instantiate();
println(args)
for arg in args; include(arg); end
println("Loading questions ...")
Knet.seed!(11131994)
trnqstns = getQdata(o[:dhome],"train")
valqstns = getQdata(o[:dhome],"val")
testdevqstns = getQdata(o[:dhome],"testdev")
println("Loading dictionaries ... ")
_,_,id2index = getDicts(o[:dhome],"dic")
dwe = load(o[:dhome]*"wordembeddings_vocabs.jld2")
qvoc,avoc,embeddings = dwe["word_dict"],dwe["answer_dict"],dwe["embeddings"];
dicts = (qvoc,avoc,id2index)
sets = []
push!(sets,miniBatch(trnqstns,dicts...;B=o[:batchsize]))
push!(sets,miniBatch(valqstns,dicts...;B=10))
push!(sets,miniBatch(testdevqstns,dicts...;B=10))
trnqstns=nothing;
valqstns=nothing;
#MODEL
gpu(0)
KnetLayers.settype!(KnetArray{Float32})
@show arrtype
#if o[:mfile] !=nothing && isfile(o[:mfile])
#    M,Mrun,o = loadmodel(o[:mfile])
#else
    M    = MACNetwork(o;embed=Param(arrtype(copy(embeddings))));
    Mrun = MACNetwork(o);
#end

for (wr,wi) in zip(params(Mrun),params(M))
    wr.value[:] = wi.value[:]
end
Knet.gc()
#FEATS
feats = loadFeatures(o[:dhome];h5=o[:h5])
M,Mrun = train!(M,Mrun,sets,feats,o)
