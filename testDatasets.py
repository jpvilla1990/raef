from datasetsModule.datasets import Datasets

datasets = Datasets()
iterator = datasets.loadDataset(
    "TSB-AD-M",
    grouping = {
        "separator" : "_",
        "element" : 1,
    },
)
datasetNames = iterator.getGroupedSubdatasets()
print(datasetNames.keys())

iterator.resetIteration("063_SMD_id_7_Facility_tr_5923_1st_6506", True, 0.8)

element = iterator.iterateDataset(
    "063_SMD_id_7_Facility_tr_5923_1st_6506",
    ["0"],
    True,
    None,
)

print(element)

iterator.resetIteration("", True, 0.8, "SMD")

element = iterator.iterateDataset(
    "",
    ["0"],
    True,
    "SMD",
)

print(element)