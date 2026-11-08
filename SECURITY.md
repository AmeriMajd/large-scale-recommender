# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's **Report a
vulnerability** flow for this repository. Do not open a public issue containing
credentials, exploit details, or sensitive logs.

Include the affected component, reproduction steps, impact, and any suggested
mitigation. You should receive an acknowledgement within seven days.

## Supported versions

The latest commit on `main` is the only supported development version until the
project publishes a tagged release.

## Deployment scope

The included Compose stack is for local development. It does not configure
TLS, Redis authentication, network isolation, backups, high availability, or a
production secrets manager. Operators are responsible for adding those
controls before exposing any service outside a trusted development machine.
