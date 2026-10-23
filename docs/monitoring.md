# SmartMaintain Prometheus and Grafana

This optional monitoring stack runs separately from the inference application.
It does not build or recreate the application image or container. Prometheus
scrapes the existing host-published application at
`host.docker.internal:8081/metrics`, while Grafana reads Prometheus over the
private Compose network. The Grafana dashboard and data source are provisioned
from files in `monitoring/grafana`.

The Compose file binds the Prometheus and Grafana UIs to loopback on ports 9090
and 3000. The production application port 8081 is not published or changed by
this stack. Do not use the start commands below against production until
explicit deployment approval has been given.

## Configuration validation

From the project directory in PowerShell:

```powershell
$env:GRAFANA_ADMIN_PASSWORD = "replace-with-a-strong-local-password"
docker compose -f compose.monitoring.yml config
```

The `config` command validates Compose interpolation and configuration without
starting containers. Review the rendered configuration before proceeding.

## Run the monitoring stack after approval

The application must already be reachable from Docker Desktop at
`http://host.docker.internal:8081/metrics`. Confirm that this endpoint responds
before starting monitoring. The monitoring stack itself does not start or
restart the application.

```powershell
$env:GRAFANA_ADMIN_PASSWORD = "replace-with-a-strong-local-password"
docker compose -f compose.monitoring.yml up -d
docker compose -f compose.monitoring.yml ps
```

Check Prometheus at <http://localhost:9090/targets> and confirm the
`smartmaintain-api` target is `UP`. Check Grafana at <http://localhost:3000>,
sign in with the configured admin username and password, and open the
**SmartMaintain HTTP Monitoring** dashboard in the **SmartMaintain** folder.

The dashboard shows:

- Request rate, grouped by method and route.
- p50 and p95 latency from the request-duration histogram.
- 4xx and 5xx response rates from the request counter's status-code label.
- Prometheus scrape availability (`up`) for the API metrics endpoint.

The availability panel means that Prometheus can scrape `/metrics`; it does not
independently probe `/health`. Histogram quantiles may be empty until requests
have been observed and sufficient scrape samples have accumulated.

## Stop only the monitoring stack

When approved, this command removes the two monitoring containers and network.
It does not target or stop `smartmaintain-rul-container`. Named data volumes are
retained unless explicitly removed:

```powershell
docker compose -f compose.monitoring.yml down
```
