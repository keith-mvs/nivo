# Enterprise Naming and Structural Conventions

**Version:** 1.0.0
**Effective Date:** 2025-12-09
**Status:** Active
**Governance:** Engineering Standards Board

---

## Naming Philosophy

Consistent naming conventions are fundamental to operational excellence in distributed systems and multi-team environments. This standard establishes a uniform, predictable vocabulary across all repositories, infrastructure components, environments, and automation artifacts. By enforcing strict casing rules, controlled vocabularies, and clear semantic patterns, we reduce cognitive load, eliminate ambiguity, accelerate onboarding, and enable reliable automation. All names must be self-documenting, machine-parseable, and aligned with industry best practices.

This standard applies to all new projects immediately and MUST be adopted by existing projects undergoing restructuring. Exceptions require documented approval from the Engineering Standards Board and must include a remediation plan. Backward compatibility considerations are required when renaming existing production resources.

---

## 1. Repository Naming (Git)

### Rules

- **Casing:** MUST use lowercase kebab-case exclusively
- **Format:** `<domain>-<purpose>-<type>` where applicable
- **Clarity:** Repository name MUST clearly identify domain and purpose
- **Length:** Aim for 2-4 segments; avoid excessive verbosity
- **Prohibited:** No underscores, camelCase, PascalCase, or mixed casing

### Categories and Examples

**Application/Service Repositories:**
- `trading-engine-svc` - Core trading execution service
- `market-data-adapter` - Market data ingestion adapter
- `risk-analytics-api` - Risk analytics REST API
- `order-management-svc` - Order lifecycle management service

**Libraries and SDKs:**
- `shared-utils-lib` - Cross-project utility library
- `logging-framework-lib` - Standardized logging framework
- `trading-sdk-python` - Python SDK for trading platform
- `data-models-lib` - Shared data models and schemas

**Platform/Shared Repositories:**
- `api-gateway` - Central API gateway service
- `auth-service` - Authentication and authorization service
- `message-bus-adapter` - Message bus integration layer
- `monitoring-stack` - Observability and monitoring infrastructure

**Infrastructure Repositories:**
- `org-infra-terraform` - Organization-wide Terraform modules
- `k8s-platform-config` - Kubernetes platform configuration
- `cicd-pipelines` - Centralized CI/CD pipeline definitions
- `network-topology-tf` - Network infrastructure as code

---

## 2. Directory and Project Root Structure

### Standard Top-Level Layout

**ALL service and library repositories MUST include:**

```
<repo-root>/
├── src/                    Production source code
├── tests/                  Test suites
│   ├── unit/               Unit tests
│   └── integration/        Integration tests
├── docs/                   Documentation
├── scripts/                Automation scripts
│   ├── dev/                Development utilities
│   └── ci/                 CI/CD automation
├── infra/                  Infrastructure as code
│   ├── terraform/          Terraform modules/config
│   └── k8s/                Kubernetes manifests
├── config/                 Runtime configuration
├── build/ or dist/         Generated artifacts (git-ignored)
├── .gitignore
├── README.md
└── CLAUDE.md              Project-specific AI context
```

### Directory Definitions

**src/**
- Production source code only
- MUST follow language-specific conventions (e.g., `src/core/`, `src/api/`)
- No test code, scripts, or documentation

**tests/**
- All test code and fixtures
- MUST separate unit and integration tests
- Additional subdirs allowed: `tests/e2e/`, `tests/performance/`

**docs/**
- Architecture Decision Records (ADRs)
- API documentation
- Operational runbooks
- System architecture diagrams
- NOT for inline code documentation

**scripts/**
- Automation utilities for development and CI/CD
- `scripts/dev/` - Local development tools (setup, benchmarks, analysis)
- `scripts/ci/` - CI/CD pipeline scripts (build, deploy, test)
- All scripts MUST be executable and include usage documentation

**infra/**
- Infrastructure as code and deployment manifests
- `infra/terraform/` - Terraform modules and configuration
- `infra/k8s/` - Kubernetes manifests and Helm charts
- Environment-specific configs in subdirectories

**config/**
- Runtime configuration files and templates
- Environment-specific configs use: `<name>.<env>.yaml`
- Templates use: `<name>.template.yaml`
- NO secrets in plaintext

**build/ or dist/**
- Generated artifacts (compiled code, bundles, packages)
- MUST be git-ignored
- Ephemeral; recreated by build process

### Subdirectory Naming

- **Casing:** MUST use lowercase kebab-case
- **Examples:** `user-management/`, `data-pipeline/`, `ml-models/`
- **Prohibited:** `UserManagement/`, `data_pipeline/`, `MLModels/`

---

## 3. Workspace Naming (IDE / VS Code)

### Rules

- **Format:** `<domain>-<project>.code-workspace`
- **Casing:** Lowercase kebab-case
- **Scope:** One workspace per logical project or monorepo
- **Prohibited:** Environment-specific names (no `-dev.code-workspace`)

### Examples

- `trading-platform.code-workspace` - Multi-service trading platform monorepo
- `risk-analytics.code-workspace` - Risk analytics service workspace
- `market-data.code-workspace` - Market data ingestion workspace
- `shared-libraries.code-workspace` - Organization shared libraries monorepo
- `infrastructure.code-workspace` - Infrastructure and platform repos

---

## 4. Environment Naming

### Standard Environment Set

**Long Names (human-readable):**
- `dev` - Development environment (active development, frequently updated)
- `test` - Testing environment (QA, automated testing)
- `stage` - Staging environment (pre-production validation)
- `prod` - Production environment (live customer-facing)

**Optional Environments:**
- `sandbox` - Isolated experimentation environment
- `perf` - Performance testing and benchmarking

**Short Codes (automation, namespaces, resource naming):**
- `dv` - Development
- `ts` - Test
- `st` - Stage
- `pd` - Production
- `sb` - Sandbox
- `pf` - Performance

### Usage Contexts

**Kubernetes Namespaces:**
- Pattern: `<service>-<env-short>`
- Examples: `trading-svc-dv`, `risk-api-st`, `auth-service-pd`

**Terraform Workspaces:**
- Pattern: `<env-short>` or `<region>-<env-short>`
- Examples: `dv`, `us-east-st`, `eu-west-pd`

**Configuration Files:**
- Pattern: `<name>.<env>.yaml`
- Examples: `appsettings.dev.yaml`, `database.prod.yaml`, `logging.stage.yaml`

**Cloud Resource Tags/Labels:**
- Key: `environment`
- Values: `dev`, `test`, `stage`, `prod`

**DNS and Hostnames:**
- Pattern: `<service>.<env>.domain.com`
- Examples: `api.dev.company.com`, `trading.prod.company.com`

---

## 5. Branches and Tags

### Branch Naming Rules

**Feature Branches:**
- Pattern: `feature/<short-description>`
- Examples: `feature/add-order-validation`, `feature/redis-caching`
- Use: New features or enhancements

**Bug Fix Branches:**
- Pattern: `fix/<short-description>`
- Examples: `fix/null-pointer-check`, `fix/memory-leak`
- Use: Non-critical bug fixes

**Hotfix Branches:**
- Pattern: `hotfix/<ticket-id>` or `hotfix/<critical-issue>`
- Examples: `hotfix/JIRA-1234`, `hotfix/production-outage`
- Use: Critical production issues requiring immediate deployment

**Release Branches:**
- Pattern: `release/<version>`
- Examples: `release/1.2.0`, `release/2.0.0-rc1`
- Use: Release preparation and stabilization

**Additional Patterns:**
- `chore/<description>` - Maintenance tasks (dependency updates, tooling)
- `docs/<description>` - Documentation-only changes
- `refactor/<description>` - Code refactoring without behavior change

### Tag Naming Rules

**Semantic Versioning (REQUIRED):**
- Pattern: `v<MAJOR>.<MINOR>.<PATCH>`
- Examples: `v1.0.0`, `v2.3.1`, `v0.1.0-beta`
- Use: All release tags

**Pre-release Tags:**
- Pattern: `v<MAJOR>.<MINOR>.<PATCH>-<pre-release>`
- Examples: `v2.0.0-alpha`, `v1.5.0-beta.2`, `v3.0.0-rc1`

**Build Metadata (optional):**
- Pattern: `v<MAJOR>.<MINOR>.<PATCH>+<build>`
- Example: `v1.0.0+20250109.abcd123`

---

## 6. Configuration File Naming

### Templates

**Purpose:** Version-controlled configuration skeletons with placeholders

**Pattern:** `<name>.template.<ext>`

**Examples:**
- `appsettings.template.yaml` - Application settings template
- `database.template.json` - Database connection template
- `secrets.template.env` - Environment variables template
- `k8s-deployment.template.yaml` - Kubernetes deployment template

**Requirements:**
- MUST NOT contain actual secrets or credentials
- MUST include placeholder documentation
- MUST be committed to version control

### Environment-Specific Configurations

**Pattern:** `<name>.<env>.<ext>`

**Examples:**
- `appsettings.dev.yaml` - Development app settings
- `database.prod.json` - Production database config
- `logging.stage.yaml` - Staging logging config
- `feature-flags.test.json` - Test environment feature flags
- `api-endpoints.prod.yaml` - Production API endpoints

**Requirements:**
- MUST use long environment names (dev, test, stage, prod)
- Secrets MUST be externalized (use secret managers, not files)
- Production configs MUST be reviewed and approved

### Secrets Management

**Prohibited:**
- Plaintext secrets in configuration files
- Committed credentials in version control
- Hardcoded API keys or passwords

**Required:**
- Use secret management systems (AWS Secrets Manager, Azure Key Vault, HashiCorp Vault)
- Reference secrets by identifier/path in configs
- Encrypt secrets at rest when stored in files (e.g., SOPS, git-crypt)

---

## 7. Infrastructure Naming

### Terraform

**Module Naming:**
- **Pattern:** `terraform-<provider>-<resource-type>`
- **Casing:** Lowercase kebab-case
- **Examples:**
  - `terraform-aws-vpc` - AWS VPC module
  - `terraform-azure-aks` - Azure Kubernetes Service module
  - `terraform-gcp-storage` - GCP storage bucket module

**Workspace Naming:**
- **Pattern:** `<env-short>` or `<region>-<env-short>`
- **Examples:** `dv`, `st`, `pd`, `us-east-pd`, `eu-west-dv`
- **Requirement:** MUST match standard environment short codes

**Resource Naming (within Terraform):**
- **Pattern:** `<project>-<resource>-<env-short>`
- **Casing:** Follow provider conventions (kebab-case or snake_case)
- **Examples:**
  - `trading-vpc-pd` - Production VPC
  - `market_data_bucket_dv` - Development S3 bucket
  - `risk-analytics-cluster-st` - Staging EKS cluster

### Kubernetes

**Namespace Naming:**
- **Pattern:** `<service>-<env-short>`
- **Examples:** `trading-svc-dv`, `auth-service-pd`, `monitoring-st`
- **System Namespaces:** `kube-system`, `kube-public`, `default` (reserved)

**Helm Release Naming:**
- **Pattern:** `<service>` (environment context from namespace)
- **Examples:** `trading-engine`, `market-data-adapter`, `api-gateway`
- **Requirement:** Release name MUST match service name

**Resource Labels (REQUIRED):**
```yaml
labels:
  app: trading-engine
  environment: prod
  version: 1.2.3
  component: backend
  team: trading-platform
```

**Deployment/Service Naming:**
- **Pattern:** `<service>-<component>` (if multi-component)
- **Examples:** `trading-engine`, `risk-api-backend`, `auth-redis`

### Cloud Resources

**AWS Naming:**
- **Casing:** kebab-case preferred
- **Pattern:** `<project>-<resource>-<env-short>`
- **Examples:**
  - `trading-engine-pd` - EC2 instance or ECS service
  - `market-data-queue-dv` - SQS queue
  - `risk-analytics-db-st` - RDS instance

**Azure Naming:**
- **Casing:** kebab-case
- **Pattern:** `<project>-<resource>-<env-short>`
- **Examples:**
  - `trading-platform-rg-pd` - Resource group
  - `market-data-storage-dv` - Storage account
  - `risk-analytics-aks-st` - AKS cluster

**GCP Naming:**
- **Casing:** kebab-case
- **Pattern:** `<project>-<resource>-<env-short>`
- **Examples:**
  - `trading-engine-vm-pd` - Compute Engine instance
  - `market-data-bucket-dv` - Cloud Storage bucket
  - `risk-analytics-gke-st` - GKE cluster

**Resource Tags/Labels (REQUIRED for all cloud resources):**
```
environment: prod
project: trading-platform
team: platform-engineering
cost-center: engineering
managed-by: terraform
```

---

## Governance and Enforcement

### Compliance Requirements

- **All new projects:** MUST comply with this standard from inception
- **Existing projects:** MUST adopt standard during restructuring or major refactoring
- **Exceptions:** Require documented approval from Engineering Standards Board
- **Backward compatibility:** MUST be assessed before renaming production resources

### Code Review Checklist

Reviewers MUST verify:
- [ ] Repository name follows lowercase kebab-case
- [ ] Directory structure matches standard layout
- [ ] Branch name follows prescribed patterns
- [ ] Configuration files use correct naming conventions
- [ ] Infrastructure resources follow naming rules
- [ ] No plaintext secrets in committed files

### Automation and CI/CD

**CI Pipelines SHOULD:**
- Validate branch names against allowed patterns
- Lint configuration file names
- Check for committed secrets (using tools like git-secrets, trufflehog)
- Verify resource naming in infrastructure code

**Automated Checks:**
- Pre-commit hooks for branch name validation
- CI stage for naming convention validation
- Terraform/Kubernetes manifest linters configured with naming rules

### Exception Process

**When exceptions are necessary:**
1. Document technical justification
2. Identify specific rules requiring exception
3. Submit request to Engineering Standards Board
4. Include remediation timeline if temporary exception
5. Maintain exception registry for audit purposes

---

## Summary Cheat Sheet

### Environment Names

| Environment | Long Name | Short Code | Use Case |
|------------|-----------|------------|----------|
| Development | dev | dv | Active development |
| Test | test | ts | QA and automated testing |
| Staging | stage | st | Pre-production validation |
| Production | prod | pd | Live customer-facing |
| Sandbox | sandbox | sb | Experimentation (optional) |
| Performance | perf | pf | Performance testing (optional) |

### Repository Naming

- **Casing:** lowercase kebab-case
- **Pattern:** `<domain>-<purpose>-<type>`
- **Examples:** `trading-engine-svc`, `shared-utils-lib`, `org-infra-terraform`

### Directory Structure

```
Standard Layout:
src/         - Production code
tests/       - Test suites (unit/, integration/)
docs/        - Documentation and ADRs
scripts/     - Automation (dev/, ci/)
infra/       - Infrastructure (terraform/, k8s/)
config/      - Runtime configuration
build/       - Generated artifacts (git-ignored)
```

### Branch Naming

| Type | Pattern | Example |
|------|---------|---------|
| Feature | feature/<desc> | feature/add-caching |
| Fix | fix/<desc> | fix/memory-leak |
| Hotfix | hotfix/<ticket> | hotfix/JIRA-1234 |
| Release | release/<version> | release/1.2.0 |
| Chore | chore/<desc> | chore/update-deps |
| Docs | docs/<desc> | docs/api-guide |

### Tag Naming

- **Pattern:** `v<MAJOR>.<MINOR>.<PATCH>`
- **Examples:** `v1.0.0`, `v2.3.1`, `v1.0.0-beta`

### Workspace Naming

- **Pattern:** `<domain>-<project>.code-workspace`
- **Examples:** `trading-platform.code-workspace`, `risk-analytics.code-workspace`

### Configuration Files

| Type | Pattern | Example |
|------|---------|---------|
| Template | `<name>.template.<ext>` | appsettings.template.yaml |
| Environment | `<name>.<env>.<ext>` | database.prod.yaml |
| Secrets | External only | AWS Secrets Manager, Vault |

### Infrastructure Naming

**Terraform:**
- Modules: `terraform-<provider>-<resource>`
- Workspaces: `<env-short>` or `<region>-<env-short>`
- Resources: `<project>-<resource>-<env-short>`

**Kubernetes:**
- Namespace: `<service>-<env-short>`
- Release: `<service>`
- Deployments: `<service>` or `<service>-<component>`

**Cloud Resources:**
- Pattern: `<project>-<resource>-<env-short>`
- Casing: kebab-case (or provider-specific)
- Tags: MUST include environment, project, team

### Quick Reference Rules

1. **Always lowercase kebab-case** for repos, branches, directories, infrastructure
2. **Use short codes** (dv, ts, st, pd) in resource names and namespaces
3. **Use long names** (dev, test, stage, prod) in config files and tags
4. **Semantic versioning** for all release tags
5. **No secrets** in version control - use secret managers
6. **Standard directory layout** for all repos
7. **Document exceptions** - obtain approval before deviating

---

**Document Control:**
- **Maintained By:** Platform Engineering Team
- **Review Cycle:** Quarterly
- **Last Reviewed:** 2025-12-09
- **Next Review:** 2026-03-09
- **Feedback:** Submit issues to engineering-standards repo
