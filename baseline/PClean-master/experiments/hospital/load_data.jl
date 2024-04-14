using CSV
using DataFrames: DataFrame

# include("/home/zj/BClean/baseline/PClean-master/src/utils.jl")

print("loading dataset\n")
dataset = "hospital"
dirty_table = CSV.File("datasets/$(dataset)_dirty1.csv") |> DataFrame
clean_table = CSV.File("datasets/$(dataset)_clean1.csv") |> DataFrame

# dirty_table = CSV.File("/home/zj/BClean/baseline/PClean-master/datasets/hospital_dirty1.csv") |> DataFrame
# clean_table = CSV.File("/home/zj/BClean/baseline/PClean-master/datasets/hospital_clean1.csv") |> DataFrame

# In the dirty data, CSV.jl infers that PhoneNumber, ZipCode, and ProviderNumber
# are strings. Our PClean script also models these columns as string-valued.
# However, in the clean CSV file (without typos) it infers they are
# numbers. To facilitate comparison of PClean's results (strings) with 
# ground-truth, we preprocess the clean values to convert them into strings.
print("preprocessing data\n")
clean_table[!, :PhoneNumber] = map(x -> "$x", clean_table[!, :PhoneNumber])
clean_table[!, :ZipCode] = map(x -> "$x", clean_table[!, :ZipCode])
clean_table[!, :ProviderNumber] = map(x -> "$x", clean_table[!, :ProviderNumber])

# Stores sets of unique observed values of each attribute.
print("storing sets\n")
possibilities = Dict(col => remove_missing(unique(collect(dirty_table[!, col])))
                     for col in propertynames(dirty_table))
print("loaded data!\n")