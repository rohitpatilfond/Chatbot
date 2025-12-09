# Chatbot
# AWS PetCare Conversational AI Lambda

Serverless Backend for Premium PetCare Chat Assistant

## Overview

This repository contains the AWS Lambda function responsible for handling conversational AI requests for the PetCare system. The Lambda integrates with Amazon Bedrock, Amazon Athena, Amazon OpenSearch, DynamoDB, and AWS Systems Manager Parameter Store to generate intelligent responses for premium users.

The system supports document search, SQL query generation for pet activity insights, chat history retrieval, and contextual response generation using pet- and user-specific details.

---

## Key Features

### 1. Configuration Management

* Retrieves all required parameters from AWS Systems Manager Parameter Store
* Loads prompt templates dynamically from Amazon S3
* Supports flexible runtime configuration such as model IDs, prompt paths, temperature, token limits, OpenSearch settings, etc.

### 2. Tooling Framework

The system supports the following tools executed through Bedrock tool-use:

#### a. Chat History Retrieval

Fetches past conversation history for a session from DynamoDB.

#### b. SQL Query Generator

Uses Amazon Bedrock to generate pet-specific Athena queries for activity analysis (walk, play, sleep, rest).

#### c. Document Search

Generates embeddings using Bedrock and searches OpenSearch for relevant document content.

---

## Architecture Components

### Amazon Bedrock

Used for:

* SQL query generation
* Activity analysis
* Natural language response generation
* Embedding generation

### Amazon Athena

Used for:

* Fetching pet activity data
* Retrieving pet names for validation
* Running SQL queries on S3-backed datasets

### Amazon OpenSearch

Used for:

* Vector search using embeddings
* Document content retrieval

### DynamoDB

Used for:

* Storing chat history
* Providing session-based conversation context

### Amazon S3

Used for:

* Storing prompt templates
* Athena output locations

### AWS Parameter Store

Used for securely storing and loading:

* API endpoints
* Model IDs
* Table names
* Prompt locations
* Guardrail identifiers
* Token limits

---

## Function Highlights

### 1. Activity Query Handling

* Extracts activity type from user input
* Builds SQL queries using Bedrock
* Validates query structure
* Runs Athena queries
* Formats the results using an LLM-based prompt

### 2. Smart Pet Name Validation

The lambda validates the pet mentioned in the query against:

* Pet name retrieved from Athena
* Prevents cross-pet querying
* Provides corrective guidance if mismatch occurs

### 3. Chat History Logging and Retrieval

* Conversations stored per session
* Maintains chronological order
* Retrieves last three interactions for context

### 4. Intelligent Query Routing

Automatically detects if the query is:

* A greeting
* A chat history request
* An activity analysis request
* A general pet-care question
  Routes the query to the appropriate internal tool.

---

## Lambda Handler Workflow

1. Load configuration (Parameter Store, S3 prompts)
2. Validate user input
3. Validate user–pet relationship via Athena
4. Load chat history from DynamoDB
5. Build context for the query
6. Route query to the appropriate tool
7. Generate response using Bedrock
8. Store interaction in DynamoDB
9. Return formatted response

---

## Folder Structure

```
lambda_function.py     # Main Lambda function
README.md              # Documentation
```

---

## Environment Requirements

### IAM Permissions

The Lambda requires permissions for:

* bedrock-runtime
* ssm:GetParameter
* athena:*
* s3:GetObject, PutObject
* dynamodb:*
* aoss:*
* logs:CreateLogGroup, CreateLogStream, PutLogEvents

### Python Libraries

* boto3
* opensearch-py
* AWSV4SignerAuth
* re, json, time, logging

These can be packaged as a Lambda layer if needed.

---

## Error Handling

The function includes:

* Graceful fallbacks
* Captured exceptions for Bedrock throttling
* Validation for SQL structure
* Timeouts for Athena queries
* User-friendly failure responses

---

## Usage

The Lambda expects the following JSON input:

```
{
  "UserId": "123",
  "PetId": "45",
  "Querry": "How much did Bruno walk yesterday?",
  "SessionId": "abc123"
}
```

Returns:

```
{
  "response": "...",
  "sources": [...]
}
```

---



https://drive.google.com/file/d/1sfTePQrNEAPTkGD20dhSlhLvVfXfRy9K/view?usp=drive_link
