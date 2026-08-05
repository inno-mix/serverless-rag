"""CloudWatch dashboard, alarms, and the monthly budget guardrail (T3.2).

Covers the three things ARCHITECTURE.md §8 asks for: a dashboard showing Lambda
health, API traffic, DLQ depth and Bedrock token usage; alarms for the two
conditions worth waking up for; and a $5 monthly budget so a runaway demo is
capped by something other than vigilance (PRD NFR-5).

Alarm and budget notifications are delivered only when an `alertEmail` is set in
`cdk.json` context. Everything else is created regardless, so the guardrails
exist even in a clone that has not configured an address.
"""

import aws_cdk as cdk
from aws_cdk import aws_apigateway as apigateway
from aws_cdk import aws_budgets as budgets
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cloudwatch_actions as cloudwatch_actions
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subscriptions
from aws_cdk import aws_sqs as sqs
from constructs import Construct

#: Monthly spend ceiling in USD (PRD NFR-5).
MONTHLY_BUDGET_USD = 5

#: Fraction of actual spend that triggers the budget notification. 80% gives
#: warning while there is still room, rather than after the fact.
BUDGET_ALERT_THRESHOLD_PERCENT = 80

#: Lambda error-rate alarm threshold, as a percentage of invocations (§8).
ERROR_RATE_THRESHOLD_PERCENT = 5

#: Custom metric namespace written by `StructuredLogger.metrics` in
#: `shared/logging_utils.py`. Must match `METRIC_NAMESPACE` there.
METRIC_NAMESPACE = "ServerlessRag"

_PERIOD = cdk.Duration.minutes(5)


class ObservabilityStack(cdk.Stack):
    """Dashboard, alarms, and budget for the whole application."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env: cdk.Environment,
        functions: dict[str, lambda_.IFunction],
        api: apigateway.RestApi,
        ingest_dlq: sqs.IQueue,
        alert_email: str = "",
    ) -> None:
        """Initialize the observability stack.

        Args:
            scope: The parent construct.
            construct_id: Stack identifier.
            env: Target account and region.
            functions: Display name to function, for the per-function widgets.
            api: The REST API whose 4xx/5xx counts are graphed.
            ingest_dlq: Dead-letter queue whose depth is alarmed on.
            alert_email: Address subscribed to alarms and the budget. Empty
                leaves the alarms in place with no notification target.
        """
        super().__init__(scope, construct_id, env=env)

        self.alarm_topic = sns.Topic(self, "AlarmTopic", display_name="serverless-rag alarms")
        if alert_email:
            self.alarm_topic.add_subscription(subscriptions.EmailSubscription(alert_email))

        self.alarms = [
            self._dlq_alarm(ingest_dlq),
            self._error_rate_alarm(functions),
        ]
        for alarm in self.alarms:
            alarm.add_alarm_action(cloudwatch_actions.SnsAction(self.alarm_topic))

        self.budget = self._budget(alert_email)
        self.dashboard = self._dashboard(functions, api, ingest_dlq)

        cdk.CfnOutput(self, "DashboardName", value=self.dashboard.dashboard_name)
        cdk.CfnOutput(self, "AlarmTopicArn", value=self.alarm_topic.topic_arn)

    # --- Alarms ------------------------------------------------------------

    def _dlq_alarm(self, ingest_dlq: sqs.IQueue) -> cloudwatch.Alarm:
        """Alarm as soon as anything lands in the dead-letter queue.

        A single message means a document failed to index, which is invisible to
        the user beyond a FAILED status, so the threshold is zero rather than a
        rate.
        """
        return cloudwatch.Alarm(
            self,
            "IngestDlqNotEmpty",
            alarm_description="A document failed to ingest and was dead-lettered",
            metric=ingest_dlq.metric_approximate_number_of_messages_visible(
                period=_PERIOD, statistic="Maximum"
            ),
            threshold=0,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )

    def _error_rate_alarm(self, functions: dict[str, lambda_.IFunction]) -> cloudwatch.Alarm:
        """Alarm when errors exceed 5% of invocations across all functions.

        Expressed as a ratio rather than a count so that a quiet period with two
        errors pages, while a busy one with two errors does not.
        """
        errors = cloudwatch.MathExpression(
            expression=" + ".join(f"e{index}" for index in range(len(functions))),
            using_metrics={
                f"e{index}": function.metric_errors(period=_PERIOD, statistic="Sum")
                for index, function in enumerate(functions.values())
            },
            period=_PERIOD,
        )
        invocations = cloudwatch.MathExpression(
            expression=" + ".join(f"i{index}" for index in range(len(functions))),
            using_metrics={
                f"i{index}": function.metric_invocations(period=_PERIOD, statistic="Sum")
                for index, function in enumerate(functions.values())
            },
            period=_PERIOD,
        )

        return cloudwatch.Alarm(
            self,
            "LambdaErrorRate",
            alarm_description=f"Lambda errors above {ERROR_RATE_THRESHOLD_PERCENT}% of invocations",
            metric=cloudwatch.MathExpression(
                # IF guards the idle case: with no invocations the ratio would be
                # 0/0, and CloudWatch would treat the resulting gap as breaching.
                expression="IF(invocations > 0, 100 * errors / invocations, 0)",
                using_metrics={"errors": errors, "invocations": invocations},
                label="Error rate (%)",
                period=_PERIOD,
            ),
            threshold=ERROR_RATE_THRESHOLD_PERCENT,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )

    # --- Budget ------------------------------------------------------------

    def _budget(self, alert_email: str) -> budgets.CfnBudget:
        """Create the monthly cost budget (PRD NFR-5).

        Budgets notify by email directly rather than through SNS, so this is the
        one alert that does not use the alarm topic.
        """
        subscribers = (
            [
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        comparison_operator="GREATER_THAN",
                        notification_type="ACTUAL",
                        threshold=BUDGET_ALERT_THRESHOLD_PERCENT,
                        threshold_type="PERCENTAGE",
                    ),
                    subscribers=[
                        budgets.CfnBudget.SubscriberProperty(
                            address=alert_email, subscription_type="EMAIL"
                        )
                    ],
                )
            ]
            if alert_email
            else None
        )

        return budgets.CfnBudget(
            self,
            "MonthlyBudget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(amount=MONTHLY_BUDGET_USD, unit="USD"),
            ),
            notifications_with_subscribers=subscribers,
        )

    # --- Dashboard ---------------------------------------------------------

    def _dashboard(
        self,
        functions: dict[str, lambda_.IFunction],
        api: apigateway.RestApi,
        ingest_dlq: sqs.IQueue,
    ) -> cloudwatch.Dashboard:
        """Build the operational dashboard described in ARCHITECTURE.md §8."""
        dashboard = cloudwatch.Dashboard(
            self,
            "RagDashboard",
            dashboard_name="serverless-rag",
            default_interval=cdk.Duration.hours(12),
        )

        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Lambda invocations",
                left=[
                    function.metric_invocations(period=_PERIOD, statistic="Sum", label=name)
                    for name, function in functions.items()
                ],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="Lambda errors",
                left=[
                    function.metric_errors(period=_PERIOD, statistic="Sum", label=name)
                    for name, function in functions.items()
                ],
                width=12,
            ),
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Lambda duration (p95)",
                left=[
                    function.metric_duration(period=_PERIOD, statistic="p95", label=name)
                    for name, function in functions.items()
                ],
                left_y_axis=cloudwatch.YAxisProps(label="ms", show_units=False),
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="API Gateway errors",
                left=[
                    api.metric_client_error(period=_PERIOD, statistic="Sum", label="4xx"),
                    api.metric_server_error(period=_PERIOD, statistic="Sum", label="5xx"),
                ],
                width=12,
            ),
        )
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Bedrock tokens (EMF)",
                left=[
                    self._token_metric("TokensIn"),
                    self._token_metric("TokensOut"),
                ],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="Ingest dead-letter queue",
                left=[
                    ingest_dlq.metric_approximate_number_of_messages_visible(
                        period=_PERIOD, statistic="Maximum", label="Messages"
                    )
                ],
                width=12,
            ),
        )
        return dashboard

    def _token_metric(self, name: str) -> cloudwatch.Metric:
        """A custom token metric published by the query Lambda through EMF."""
        return cloudwatch.Metric(
            namespace=METRIC_NAMESPACE,
            metric_name=name,
            dimensions_map={"Service": "query"},
            period=_PERIOD,
            statistic="Sum",
            label=name,
        )
