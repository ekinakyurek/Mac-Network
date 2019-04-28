initEmbRandom(embDim,InputDim) =  randn(Float32,embDim,InputDim)

function sentenceEmb(sentence, wordVectors, dim)
    words = split(sentence)
    wordEmbs = initEmbRandom(dim, length(words))
    for (idx, word) in enumerate(words)
        if haskey(wordVectors,word)
            wordEmbs[:,idx] .= wordVectors[word]
        end
    end
    return vec(sum(wordEmbs,dims=2))
end

function initializeWordEmbeddings(dim; wordsDict = nothing, prefix = "data/GQA/glove.6B.")
    # default dictionary to use for embeddings
    embeddings = initEmbRandom(dim,length(wordsDict))
    wordVectors  = Dict()
    open(prefix*string(dim)*"d.txt") do f
        for line in eachline(f)
            line = split(strip(line))
            word = lowercase(line[1])
            vector = [parse(Float32,x) for x in line[2:end]]
            wordVectors[word] = vector
        end
    end
    

    for (w,index) in wordsDict
        if occursin(" ",w)
            print("he")
            embeddings[:,index] .= sentenceEmb(w, wordVectors)
        else
            if haskey(wordVectors,w)
                embeddings[:,index] .= wordVectors[w]
            end
        end
    end
    return embeddings                  
end
