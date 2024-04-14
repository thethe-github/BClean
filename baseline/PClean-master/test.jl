using PClean
using DataFrames: DataFrame
import CSV

# Load data
dataset = "hospital"
dirty_table = CSV.File("datasets/$(dataset)_dirty2.csv") |> DataFrame
clean_table = CSV.File("datasets/$(dataset)_clean2.csv") |> DataFrame

clean_table[!, :PhoneNumber] = map(x -> "$x", clean_table[!, :PhoneNumber])
clean_table[!, :ZipCode] = map(x -> "$x", clean_table[!, :ZipCode])
clean_table[!, :ProviderNumber] = map(x -> "$x", clean_table[!, :ProviderNumber])

# Stores sets of unique observed values of each attribute.
possibilities = Dict(col => remove_missing(unique(collect(dirty_table[!, col])))
                     for col in propertynames(dirty_table))
print("load data!\n")

# Define PClean model
PClean.@model HospitalModel begin 
    @class County begin
        @learned state_proportions::ProportionsParameter
        state ~ ChooseProportionally(possibilities[:State], state_proportions)
        county ~ StringPrior(3, 30, possibilities[:CountyName])
    end;
    @class Place begin
        county ~ County
        city ~ StringPrior(3, 30, possibilities[:City])
    end;
    @class Condition begin
        desc ~ StringPrior(5, 35, possibilities[:Condition])
    end;
    @class Measure begin
        code ~ ChooseUniformly(possibilities[:MeasureCode])
        name ~ ChooseUniformly(possibilities[:MeasureName])
        condition ~ Condition
    end;
    @class HospitalType begin
        desc ~ StringPrior(10, 30, possibilities[:HospitalType])
    end;
    @class Hospital begin
        @learned owner_dist::ProportionsParameter
        @learned service_dist::ProportionsParameter
        loc ~ Place; type ~ HospitalType
        provider ~ ChooseUniformly(possibilities[:ProviderNumber])
        name ~ StringPrior(3, 50, possibilities[:HospitalName])
        addr ~ StringPrior(10, 30, possibilities[:Address1])
        phone ~ StringPrior(10, 10, possibilities[:PhoneNumber])
        owner ~ ChooseProportionally(possibilities[:HospitalOwner], owner_dist)
        zip ~ ChooseUniformly(possibilities[:ZipCode])
        service ~ ChooseProportionally(possibilities[:EmergencyService], service_dist)
    end;
    @class Record begin
        begin
            hosp     ~ Hospital;                         service ~ AddTypos(hosp.service)
            provider ~ AddTypos(hosp.provider);          name    ~ AddTypos(hosp.name)
            addr     ~ AddTypos(hosp.addr);              city    ~ AddTypos(hosp.loc.city)
            state    ~ AddTypos(hosp.loc.county.state);  zip     ~ AddTypos(hosp.zip)
            county   ~ AddTypos(hosp.loc.county.county); phone   ~ AddTypos(hosp.phone)
            type     ~ AddTypos(hosp.type.desc);         owner   ~ AddTypos(hosp.owner)
        end
        begin
            metric ~ Measure
            code ~ AddTypos(metric.code);
            mname ~ AddTypos(metric.name);
            condition ~ AddTypos(metric.condition.desc)
            stateavg = "$(hosp.loc.county.state)_$(metric.code)"
            stateavg_obs ~ AddTypos(stateavg)
        end
    end;
end;

# Align column names of CSV with variables in the model.
# Format is ColumnName CleanVariable DirtyVariable, or, if
# there is no corruption for a certain variable, one can omit
# the DirtyVariable.
query = @query HospitalModel.Record [
    ProviderNumber   hosp.provider          provider
    HospitalName     hosp.name              name
    HospitalType     hosp.type.desc         type
    HospitalOwner    hosp.owner             owner
    Address1         hosp.addr              addr
    PhoneNumber      hosp.phone             phone
    EmergencyService hosp.service           service
    City             hosp.loc.city          city
    CountyName       hosp.loc.county.county county
    State            hosp.loc.county.state  state
    ZipCode          hosp.zip               zip
    Condition        metric.condition.desc  condition
    MeasureCode      metric.code            code
    MeasureName      metric.name            mname
    Stateavg         stateavg               stateavg_obs
];

# Configuration
config = PClean.InferenceConfig(1, 2; use_mh_instead_of_pg=true);

observations = [ObservedDataset(query, dirty_table)];
@time begin 
    trace = initialize_trace(observations, config);
    run_inference!(trace, config);
end

# Evaluate accuracy, if ground truth is available
results = evaluate_accuracy(dirty_table, clean_table, trace.tables[:Record], query)
PClean.save_results("results", "hospital", trace, observations)

# Can print results.f1, results.precision, results.accuracy, etc.
println(results)