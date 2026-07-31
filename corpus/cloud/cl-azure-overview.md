---
doc_id: cl-azure-overview
title: Azure AI Platform Overview
domain: cloud
source_type: vendor_docs
publisher: Microsoft Azure
published: 2026-01-30
covers: [Azure]
reliability: high
synthetic: true
---

# Azure AI Platform

Azure AI Foundry provides managed access to OpenAI models alongside Microsoft and open-weight
models. Container Apps offers scale-to-zero container hosting.

## Scaling

Container Apps scales to zero. Provisioned throughput units are available for predictable
latency but must be reserved monthly.

## Data residency

Azure OpenAI is available in 24 regions. Enterprise agreements can pin data processing to a
named geography, which is frequently the deciding factor for regulated customers.

## Cost structure

Pay-as-you-go token pricing, or provisioned throughput units billed monthly regardless of use.
Egress is 0.087 USD per GB.

## Operational burden

Azure's portal experience is well regarded, but quota approval for OpenAI models can take several
business days, which affects project timelines. Teams already using Microsoft identity
infrastructure report the lowest integration effort of the three major clouds.
