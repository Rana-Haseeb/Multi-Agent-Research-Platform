---
doc_id: cl-aws-overview
title: Deploying AI Workloads on AWS
domain: cloud
source_type: vendor_docs
publisher: Amazon Web Services
published: 2026-02-27
covers: [AWS]
reliability: high
synthetic: true
---

# Deploying AI Workloads on AWS

AWS offers managed inference through Bedrock, container hosting through ECS and EKS, and
serverless execution through Lambda. Bedrock provides access to third-party foundation models
without managing GPU capacity.

## Scaling

ECS Fargate scales to zero is not supported; a minimum task count must remain running. EKS
supports cluster autoscaling but requires operating the control plane configuration.

## Data residency

Bedrock is available in 14 regions. Model availability differs by region, and several models are
restricted to us-east-1 and us-west-2.

## Cost structure

Charges are per input and output token for Bedrock, plus compute for any hosted component. Egress
is billed separately at 0.09 USD per GB after the first 100 GB monthly.

## Operational burden

AWS requires the most configuration of the major providers - IAM policies, VPC networking and
security groups must be set up before a service is reachable. Teams without a platform engineer
commonly report multi-day setup for a first production deployment.
