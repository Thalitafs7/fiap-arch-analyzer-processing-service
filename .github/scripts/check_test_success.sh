#!/bin/bash
set -e

echo "Starting Terraform checks..."

# 1. Terraform Init
echo "Running: terraform init"
terraform init -backend=false

# 2. Terraform Format Check
echo "Running: terraform fmt -check"
terraform fmt -check -recursive

# 3. Terraform Validate
echo "Running: terraform validate"
terraform validate

echo "All Terraform checks passed successfully!"
