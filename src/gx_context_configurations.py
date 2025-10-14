import great_expectations as gx
from great_expectations.core.batch import BatchRequest
from datasets import load_from_disk
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, PROJ_ROOT

DATASOURCE_NAME = "data_check"
RAW_TRAIN_DATA = "raw_train"
CLEAN_TRAIN_DATA = "clean_train"
BATCH_RAW_TRAIN_DEFINITION = "batch_raw_train"
BATCH_CLEAN_TRAIN_DEFINITION = "batch_clean_train"
RAW_TRAIN_VALIDATOR = "raw_train_validator"
CLEAN_TRAIN_VALIDATOR = "clean_train_validator"
RAW_VAL_DATA = "raw_val"
CLEAN_VAL_DATA = "clean_val"
BATCH_RAW_VAL_DEFINITION = "batch_raw_val"
BATCH_CLEAN_VAL_DEFITNITION = "batch_clean_val"
RAW_VAL_VALIDATOR = "raw_val_validator"
CLEAN_VAL_VALIDATOR = "clean_val_validator"
RAW_TEST_DATA = "raw_test"
BATCH_RAW_TEST_DEFINITION = "batch_raw_test"
BATCH_CLEAN_TEST_DEFINITION = "batch_clean_test"
CLEAN_TEST_DATA = "clean_test"
RAW_TEST_VALIDATOR = "raw_test_validator"
CLEAN_TEST_VALIDATOR = "clean_test_validator"
EXPECTATIONS_SUITE = "data_quality_validation"
CHECKPOINT = "validation_checkpoint"

if __name__ == "__main__":
    context = gx.get_context(mode="file", project_root_dir = PROJ_ROOT)

    data_docs_config = {
        "class_name": "SiteBuilder",
        "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
        "store_backend": {
            "class_name": "TupleFilesystemStoreBackend",
            "base_directory": "data_docs", 
        },
    }
    context.update_data_docs_site("local_site", data_docs_config)

    datasource = context.data_sources.add_or_update_pandas(name=DATASOURCE_NAME)

    raw_train = datasource.add_parquet_asset(name=RAW_TRAIN_DATA, path=RAW_DATA_DIR/"train.parquet")
    raw_train_batch_definition = raw_train.add_batch_definition(name=BATCH_RAW_TRAIN_DEFINITION)

    raw_val = datasource.add_parquet_asset(name=RAW_VAL_DATA, path=RAW_DATA_DIR/"validation.parquet")
    raw_val_batch_definition = raw_val.add_batch_definition(name=BATCH_RAW_VAL_DEFINITION)

    raw_test = datasource.add_parquet_asset(name=RAW_TEST_DATA, path=RAW_DATA_DIR/"test.parquet")
    raw_test_batch_definition = raw_test.add_batch_definition(name=BATCH_RAW_TEST_DEFINITION)

    clean_train = datasource.add_parquet_asset(name=CLEAN_TRAIN_DATA, path=PROCESSED_DATA_DIR/"clean_train.parquet")
    clean_train_batch_definition = clean_train.add_batch_definition(name=BATCH_CLEAN_TRAIN_DEFINITION)

    clean_val = datasource.add_parquet_asset(name=CLEAN_VAL_DATA, path=PROCESSED_DATA_DIR/"clean_val.parquet")
    clean_val_batch_definition = clean_val.add_batch_definition(name=BATCH_CLEAN_VAL_DEFITNITION)

    clean_test = datasource.add_parquet_asset(name=CLEAN_TEST_DATA, path=PROCESSED_DATA_DIR/"clean_test.parquet")
    clean_test_batch_definition = clean_test.add_batch_definition(name=BATCH_CLEAN_TEST_DEFINITION)

    expectation_suite = gx.ExpectationSuite(EXPECTATIONS_SUITE)
    context.suites.add_or_update(expectation_suite)

    # Validate of column id
    expectation_suite.add_expectation(gx.expectations.ExpectColumnToExist(column="id"))
    expectation_suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(column="id", type_="str"))
    expectation_suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="id"))
    expectation_suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="id"))

    # Validate of column article
    expectation_suite.add_expectation(gx.expectations.ExpectColumnToExist(column="article"))
    expectation_suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(column="article", type_="str"))
    expectation_suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="article"))
    expectation_suite.add_expectation(gx.expectations.ExpectColumnValueLengthsToBeBetween(column='article', min_value=50, strict_min=True))

    # Validate of column highlights
    expectation_suite.add_expectation(gx.expectations.ExpectColumnToExist(column="highlights"))
    expectation_suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(column="highlights", type_="str"))
    expectation_suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="highlights"))

    # Validate of column pair-wise article-highlight
    expectation_suite.add_expectation(gx.expectations.ExpectCompoundColumnsToBeUnique(column_list=["article", "highlights"]))

    expectation_suite.save()

    raw_data_train_validation_definition = gx.ValidationDefinition(
        name=RAW_TRAIN_VALIDATOR,
        data=raw_train_batch_definition,
        suite=expectation_suite,
    )
    context.validation_definitions.add_or_update(raw_data_train_validation_definition)

    raw_data_val_validation_definition = gx.ValidationDefinition(
        name=RAW_VAL_VALIDATOR,
        data=raw_val_batch_definition,
        suite=expectation_suite
    )
    context.validation_definitions.add_or_update(raw_data_val_validation_definition)

    raw_data_test_validation_definition = gx.ValidationDefinition(
        name=RAW_TEST_VALIDATOR,
        data=raw_test_batch_definition,
        suite=expectation_suite
    )
    context.validation_definitions.add_or_update(raw_data_test_validation_definition)

    clean_data_train_validation_definition = gx.ValidationDefinition(
        name=CLEAN_TRAIN_VALIDATOR,
        data=clean_train_batch_definition,
        suite=expectation_suite
    )
    context.validation_definitions.add_or_update(clean_data_train_validation_definition)

    clean_data_val_validation_definition = gx.ValidationDefinition(
        name=CLEAN_VAL_VALIDATOR,
        data=clean_val_batch_definition,
        suite=expectation_suite
    )
    context.validation_definitions.add_or_update(clean_data_val_validation_definition)

    clean_data_test_validation_definition = gx.ValidationDefinition(
        name=CLEAN_TEST_VALIDATOR,
        data=clean_test_batch_definition,
        suite=expectation_suite
    )
    context.validation_definitions.add_or_update(clean_data_test_validation_definition)

    action_list = [
        gx.checkpoint.UpdateDataDocsAction(name="update_data_docs")
    ]

    validation_definitions = [raw_data_train_validation_definition,
                              raw_data_val_validation_definition,
                              raw_data_test_validation_definition,
                              clean_data_train_validation_definition,
                              clean_data_val_validation_definition,
                              clean_data_test_validation_definition]
    
    checkpoint = gx.Checkpoint(
        name=CHECKPOINT,
        validation_definitions=validation_definitions,
        actions=action_list,
        result_format="SUMMARY"
    )

    context.checkpoints.add_or_update(checkpoint)