o=Dict(:h5=>false,
       :mfile=>nothing,
       :epochs=>16,
       :lr=>0.0001,
       :p=>12,
       :ema=>0.999f0,
       :batchsize=>32, #x2
       :selfattn=>false,
       :gating=>false,
       :shuffle=>true,
       :sorted=>false,
       :prefix=>string(now())[1:10],
       :vocab_size=>90,
       :embed_size=>300, 
       :dhome=>"data/", 
       :loadresnet=>false,
       :d=>512)