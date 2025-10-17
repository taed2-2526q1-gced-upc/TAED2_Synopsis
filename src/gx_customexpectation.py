from typing import Optional

from great_expectations.execution_engine import (
    PandasExecutionEngine,
)
from great_expectations.expectations.expectation import ColumnPairMapExpectation
from great_expectations.expectations.expectation_configuration import (
    ExpectationConfiguration,
)
from great_expectations.expectations.metrics.map_metric_provider import (
    ColumnPairMapMetricProvider,
    column_pair_condition_partial,
)


class ColumnPairValuesColumnShorterThanColumn(ColumnPairMapMetricProvider):
    condition_metric_name = "column_pair_values.highlights_shorter_than_article"
    condition_domain_keys = (
        "column_A",
        "column_B",
    )
    condition_value_keys = ()

    @column_pair_condition_partial(engine=PandasExecutionEngine)
    def _pandas(cls, column_A, column_B, **kwargs):
        return column_A.str.len() < column_B.str.len()

class ExpectColumnPairValuesColumnShorterThanColumn(ColumnPairMapExpectation):
    """Expectation: Highlights is shorter than article"""

    examples = []

    map_metric = "column_pair_values.highlights_shorter_than_article"

    success_keys = (
        "column_A",
        "column_B",
        "mostly",
    )

    def validate_configuration(
        self, configuration: Optional[ExpectationConfiguration]
    ) -> None:

        super().validate_configuration(configuration)
        configuration = configuration or self.configuration

        library_metadata = {
            "description": "Values of column A character length must be shorter than values of column B character length"
        }



if __name__ == "__main__":
    ExpectColumnPairValuesColumnShorterThanColumn().print_diagnostic_checklist()