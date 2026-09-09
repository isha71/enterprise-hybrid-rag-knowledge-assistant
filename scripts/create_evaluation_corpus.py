#!/usr/bin/env python3
"""Generate synthetic evaluation corpus PDFs.

Creates deterministic employee_handbook.pdf and product_manual.pdf
with known content on specific pages so evaluation questions have
stable page-level relevance labels.

Usage:
    python scripts/create_evaluation_corpus.py
"""

from pathlib import Path
from fpdf import FPDF


OUTPUT_DIR = Path("evaluation/corpus")


# ─── Employee Handbook Content (8 pages) ─────────────────────────────

EMPLOYEE_HANDBOOK_PAGES = [
    # Page 1: Company Overview
    """GLOBALTECH SOLUTIONS — EMPLOYEE HANDBOOK

Company Overview

GlobalTech Solutions is a technology company founded in 2010. Our mission is to deliver innovative enterprise software solutions that empower businesses worldwide.

Company Values:
- Innovation: We encourage creative problem-solving.
- Integrity: We operate with transparency and honesty.
- Collaboration: We believe in the power of teamwork.
- Excellence: We strive for the highest quality in everything we do.

This handbook applies to all full-time and part-time employees. It outlines company policies, benefits, and expectations. All employees are expected to read and acknowledge this handbook upon joining the organization.""",

    # Page 2: Onboarding Process
    """Onboarding Process for New Hires

All new employees follow a structured onboarding program during their first two weeks.

Week 1:
- Day 1: Welcome orientation, badge issuance, IT equipment setup.
- Day 2: HR policy review and benefits enrollment.
- Day 3-5: Team introductions, role-specific training, and system access provisioning.

Week 2:
- Shadowing a senior team member.
- Completing mandatory compliance training modules.
- Meeting with direct manager to review 30/60/90 day goals.

Probation Period:
New employees are subject to a 90-day probationary period. During this time, performance is reviewed at 30-day intervals. Successful completion of the probation period leads to full employment status with access to all benefits.""",

    # Page 3: Annual Leave and Paid Time Off
    """Annual Leave and Paid Time Off Policy

New employees receive 15 days of paid time off (PTO) per year, accrued monthly. After three years of continuous service, PTO increases to 20 days per year. After seven years, employees receive 25 days per year.

PTO Requests:
- Submit PTO requests at least two weeks in advance via the HR portal.
- Requests are approved based on team capacity and business needs.
- Unused PTO may be carried over up to a maximum of 5 days into the next calendar year.

Sick Leave:
Employees are entitled to 10 days of paid sick leave per year. Sick leave does not carry over between years. A medical certificate is required for absences exceeding three consecutive days.

Public Holidays:
The company observes 11 public holidays per year as defined in the annual holiday calendar.""",

    # Page 4: Remote Work Policy
    """Remote Work and Work From Home Policy

GlobalTech Solutions supports flexible work arrangements. Eligible employees may work remotely up to three days per week with manager approval.

Eligibility:
- Employees must have completed the 90-day probation period.
- The employee's role must be compatible with remote work.
- Performance must meet or exceed expectations.

Remote Work Guidelines:
- Maintain regular working hours (9:00 AM to 5:30 PM local time).
- Be available on Slack and email during core hours (10:00 AM to 3:00 PM).
- Use the company VPN for all work-related activities.
- Ensure a stable internet connection with minimum 25 Mbps download speed.
- Attend mandatory in-office days as designated by the team lead.

Equipment:
Remote workers receive a company laptop and monitor. Employees are responsible for maintaining an ergonomic workspace at home.""",

    # Page 5: Travel and Expense Reimbursement
    """Travel and Expense Reimbursement Policy

Policy Code: HR-401

This policy governs all business-related travel and expense reimbursements for employees of GlobalTech Solutions.

Eligible Expenses:
- Airfare (economy class for domestic, premium economy for international flights over 6 hours).
- Hotel accommodations up to $200 per night for domestic travel, $300 for international.
- Ground transportation including taxis, rideshare, and rental cars.
- Meals up to $75 per day with itemized receipts.
- Conference registration fees pre-approved by management.

Reimbursement Process:
1. Submit expense reports within 14 days of travel completion.
2. Attach original receipts for all expenses exceeding $25.
3. Use the company expense management system (ExpenseTrack).
4. Reports are reviewed by the direct manager and finance team.
5. Approved reimbursements are processed within 10 business days.

Non-Reimbursable Items:
Personal entertainment, alcohol, spouse/partner travel expenses, and premium upgrades without prior approval.""",

    # Page 6: Maternity and Parental Leave
    """Maternity and Parental Leave Policy

Maternity Leave:
Female employees are entitled to 16 weeks of paid maternity leave. Leave may begin up to 4 weeks before the expected due date. An additional 4 weeks of unpaid leave may be requested.

Paternity Leave:
Male employees and partners are entitled to 4 weeks of paid paternity leave, to be taken within 6 months of the child's birth or adoption.

Adoption Leave:
Employees adopting a child under the age of 5 are entitled to 12 weeks of paid adoption leave.

Return to Work:
- Employees returning from parental leave are guaranteed the same or equivalent position.
- A gradual return schedule (part-time for up to 4 weeks) is available upon request.
- Lactation rooms are available at all office locations.

Benefits During Leave:
Health insurance and other benefits continue during paid parental leave. Employees on unpaid leave may continue benefits at their own cost.""",

    # Page 7: Security and IT Policies
    """Information Security and IT Policies

Password Requirements:
All system passwords must meet the following complexity requirements:
- Minimum 12 characters in length.
- Must contain at least one uppercase letter, one lowercase letter, one digit, and one special character.
- Passwords must be changed every 90 days.
- Previous 10 passwords cannot be reused.
- Multi-factor authentication (MFA) is mandatory for all systems.

Security Training:
All employees must complete annual cybersecurity awareness training by December 31 of each year. Training covers phishing awareness, data handling, and incident reporting procedures. Failure to complete training results in temporary access suspension.

IT Support:
For IT support issues, contact the IT Help Desk:
- Email: itsupport@globaltech.com
- Phone: ext. 4500
- Slack: #it-helpdesk
- Hours: Monday to Friday, 8:00 AM to 6:00 PM
- Emergency after-hours support: +1-555-0199

Acceptable Use:
Company devices and networks are for business purposes. Limited personal use is acceptable during non-work hours. Downloading unauthorized software is prohibited.""",

    # Page 8: Performance Reviews and Data Retention
    """Performance Review Process

Annual Performance Reviews:
Performance reviews are conducted annually during the month of March. The review cycle includes:
1. Self-assessment submission (February 1-15).
2. Peer feedback collection (February 15-28).
3. Manager evaluation and calibration (March 1-15).
4. One-on-one review meeting (March 15-31).

Rating Scale:
- Exceeds Expectations
- Meets Expectations
- Needs Improvement
- Unsatisfactory

Mid-Year Check-In:
An informal mid-year check-in occurs in September to review progress against goals.

Data Retention Policy

The company retains employee records and business data according to the following schedule:
- Active employee records: retained during employment plus 7 years.
- Financial records: retained for 10 years.
- Email communications: retained for 3 years.
- Project documentation: retained for 5 years after project completion.
- Customer data: retained per contractual obligations, minimum 5 years.

Data deletion requests are processed within 30 business days. All data retention complies with GDPR and applicable local regulations.""",
]


# ─── Product Manual Content (8 pages) ────────────────────────────────

PRODUCT_MANUAL_PAGES = [
    # Page 1: Product Overview
    """NEXUSPLATFORM XR-7000 — PRODUCT MANUAL

Product Overview

The NexusPlatform XR-7000 is an enterprise-grade data integration and analytics platform designed for large-scale organizations. Model identifier: NP-XR-7000-ENT.

Key Capabilities:
- Real-time data ingestion from 200+ source connectors.
- Built-in ETL pipeline builder with drag-and-drop interface.
- Advanced analytics engine with SQL and Python support.
- Role-based access control with SSO integration.
- Horizontal scaling across distributed clusters.

Product Editions:
- XR-7000 Standard: Up to 100 concurrent users.
- XR-7000 Professional: Up to 500 concurrent users with premium connectors.
- XR-7000 Enterprise: Unlimited users, dedicated support, custom SLA.

This manual covers installation, configuration, operation, and troubleshooting for all editions of the XR-7000 platform.""",

    # Page 2: System Requirements
    """System Requirements

Minimum Hardware Requirements:
- CPU: 8 cores (Intel Xeon or AMD EPYC recommended).
- RAM: 32 GB minimum, 64 GB recommended for production.
- Storage: 500 GB SSD with minimum 3000 IOPS.
- Network: 1 Gbps Ethernet interface.

Supported Operating Systems:
- Ubuntu 20.04 LTS or later.
- Red Hat Enterprise Linux 8.x or later.
- CentOS Stream 8 or later.
- Windows Server 2019 or later (limited support).

Software Prerequisites:
- Java Runtime Environment (JRE) 17 or later.
- PostgreSQL 14 or later (for metadata store).
- Docker 24.0 or later (for containerized deployment).
- Python 3.10 or later (for custom analytics scripts).

Browser Compatibility (Web Dashboard):
- Google Chrome 100+
- Mozilla Firefox 100+
- Microsoft Edge 100+
- Safari 15+""",

    # Page 3: Installation
    """Installation Guide

Step 1: Download the installer package from the NexusPlatform portal.
Step 2: Verify the checksum: sha256sum nexusplatform-xr7000-latest.tar.gz
Step 3: Extract the archive: tar -xzf nexusplatform-xr7000-latest.tar.gz
Step 4: Run the installer: ./install.sh --edition enterprise
Step 5: Follow the interactive setup wizard to configure database connection, admin credentials, and license key.

Initial Configuration:
After installation, access the web dashboard at https://localhost:8443 with the admin credentials created during setup.

Default Ports:
- Web Dashboard: 8443 (HTTPS)
- API Gateway: 9090
- Data Ingestion: 9091
- Metrics: 9092

License Activation:
Enter your license key in Settings > License Management. The platform operates in trial mode (30 days, 10 users) without a license key.""",

    # Page 4: API Configuration
    """API Configuration and Authentication

API Authentication Tokens:
The XR-7000 API uses bearer token authentication. To generate an API token:
1. Navigate to Settings > API Management > Tokens.
2. Click "Generate New Token."
3. Select the permission scope (read-only, read-write, or admin).
4. Set an expiration period (30, 90, 180, or 365 days).
5. Copy the generated token immediately — it will not be shown again.

API Rate Limits:
- Standard edition: 100 requests per minute per token.
- Professional edition: 500 requests per minute per token.
- Enterprise edition: 2000 requests per minute per token.

API Base URL:
https://<your-instance>/api/v2/

Example API Call:
curl -H "Authorization: Bearer <token>" https://instance.example.com/api/v2/datasets

Webhook Configuration:
Configure webhooks in Settings > Integrations > Webhooks to receive real-time notifications for data pipeline events, alerts, and system status changes.""",

    # Page 5: Error Codes and Troubleshooting
    """Error Codes and Troubleshooting

Common Error Codes:

E100 - Connection Timeout: The platform could not reach the data source within the configured timeout period. Check network connectivity and firewall rules.

E101 - Authentication Failed: Invalid credentials for the data source connector. Verify username, password, or API key.

E102 - Schema Mismatch: The source data schema does not match the expected target schema. Review field mappings in the pipeline configuration.

E103 - Storage Full: The storage volume has reached capacity. Free disk space or expand the volume.

E104 - License Expired: The platform license has expired. Contact sales@nexusplatform.com to renew. The platform continues in read-only mode for 7 days after expiration.

E105 - Rate Limit Exceeded: API rate limit has been exceeded. Wait for the rate limit window to reset or upgrade your edition.

Reset Procedure:
To reset the platform to factory defaults:
1. Stop all services: nexusctl stop --all
2. Run the reset command: nexusctl reset --factory --confirm
3. This removes all data, configurations, and user accounts.
4. Re-run the installation wizard after reset.

WARNING: Factory reset is irreversible. Create a backup before proceeding.""",

    # Page 6: Network Requirements
    """Network Requirements and Connectivity

Network Architecture:
The XR-7000 requires the following network configuration for proper operation:

Required Ports (Inbound):
- TCP 8443: Web dashboard (HTTPS).
- TCP 9090: API gateway.
- TCP 9091: Data ingestion endpoint.

Required Ports (Outbound):
- TCP 443: License server, update server, cloud connectors.
- TCP 5432: PostgreSQL metadata store (if external).
- TCP/UDP as required by configured data source connectors.

Firewall Requirements:
- Allow outbound HTTPS to license.nexusplatform.com for license validation.
- Allow outbound HTTPS to updates.nexusplatform.com for software updates.
- Internal cluster nodes must have unrestricted communication on ports 9100-9110.

Load Balancer Configuration:
For high-availability deployments, configure a Layer 7 load balancer with:
- Health check endpoint: GET /api/v2/health
- Session affinity: enabled (cookie-based).
- SSL termination at load balancer recommended.

Bandwidth Recommendations:
- Minimum: 1 Gbps for standard workloads.
- Recommended: 10 Gbps for high-throughput data ingestion scenarios.""",

    # Page 7: SLA and Warranty
    """Service Level Agreement and Warranty

SLA Uptime Guarantee:
The XR-7000 Enterprise edition includes a 99.9% uptime SLA for the managed cloud deployment option. This translates to a maximum of 8.76 hours of unplanned downtime per year. Scheduled maintenance windows are excluded from SLA calculations.

SLA Credits:
- 99.0% - 99.9% uptime: 10% service credit.
- 95.0% - 99.0% uptime: 25% service credit.
- Below 95.0% uptime: 50% service credit.

Support Response Times:
- Critical (P1): 1 hour response, 4 hour resolution target.
- High (P2): 4 hour response, 1 business day resolution target.
- Medium (P3): 1 business day response.
- Low (P4): 2 business day response.

Warranty:
The XR-7000 hardware appliance (if purchased) includes a 3-year limited warranty covering manufacturing defects. The warranty does not cover damage from improper installation, unauthorized modifications, or environmental factors. Software updates are included for the duration of the active subscription.

Extended warranty and premium support plans are available. Contact your account manager for details.""",

    # Page 8: Enterprise Subscriptions and Refund Policy
    """Enterprise Subscription Plans and Pricing

Subscription Tiers:
- Standard: $500/month — up to 100 users, 1 TB storage, email support.
- Professional: $1,500/month — up to 500 users, 5 TB storage, priority support.
- Enterprise: Custom pricing — unlimited users, unlimited storage, dedicated support, custom SLA.

All subscriptions are billed annually. Monthly billing is available at a 15% premium.

Refund Policy for Enterprise Subscriptions:
Enterprise subscription refunds are processed under the following conditions:
1. Cancellation within 30 days of initial purchase: full refund.
2. Cancellation within 31-90 days: pro-rated refund minus a 15% administrative fee.
3. Cancellation after 90 days: no refund; service continues until the end of the billing period.
4. Refund requests must be submitted in writing to billing@nexusplatform.com.
5. Custom development work and professional services fees are non-refundable.

Subscription Renewal:
Subscriptions auto-renew 30 days before expiration. Customers are notified 60 days in advance. To cancel auto-renewal, submit a request at least 45 days before the renewal date.""",
]


def create_pdf(pages: list[str], output_path: Path, title: str) -> None:
    """Create a multi-page PDF with the given text content."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    for page_text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)
        # Encode to latin-1 and replace unsupported chars
        safe_text = page_text.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 5, safe_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    print(f"Created: {output_path} ({len(pages)} pages)")


def main():
    create_pdf(
        EMPLOYEE_HANDBOOK_PAGES,
        OUTPUT_DIR / "employee_handbook.pdf",
        "GlobalTech Solutions Employee Handbook",
    )
    create_pdf(
        PRODUCT_MANUAL_PAGES,
        OUTPUT_DIR / "product_manual.pdf",
        "NexusPlatform XR-7000 Product Manual",
    )
    print(f"\nEvaluation corpus created in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
