import json
import logging
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import time
import re
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
bedrock_client = boto3.client('bedrock-runtime')
athena_client = boto3.client('athena')
ssm_client = boto3.client('ssm')
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Configuration dictionary to store values from Parameter Store
config = {}

def get_parameter(param_name):
    """Retrieve parameter from AWS Parameter Store"""
    try:
        response = ssm_client.get_parameter(
            Name=param_name,
            WithDecryption=True
        )
        return response['Parameter']['Value']
    except Exception as e:
        logger.error(f"Error retrieving parameter {param_name}: {str(e)}")
        raise

def load_config_from_parameter_store():
    """Load all configuration from Parameter Store"""
    try:
        logger.info("Loading configuration from Parameter Store")
        global config
        config = {
            "OPENSEARCH_ENDPOINT": get_parameter("rcs-fond-premium-user-opensearch-endpoint"),
            "OPENSEARCH_REGION": get_parameter("rcs-fond-premium-user-region"),
            "INDEX_NAME": get_parameter("rcs-fond-premium-user-index-name"),
            "ATHENA_DATABASE": get_parameter("rcs-fond-premium-user-athena-DB"),
            "ATHENA_OUTPUT_BUCKET": get_parameter("rcs-fond-premium-user-s3-bucket"),
            "ATHENA_TABLES": json.loads(get_parameter("rcs-fond-premium-user-athena-table")),
            "MODEL_ID": get_parameter("rcs-fond-premium-modelid"),
            "CHATHISTORY_TABLE": get_parameter("rcs-fond-chathistory-dynamodb"),
            "TEMPERATURE": float(get_parameter("rcs-fond-premium-user-temperature")),
            "MAXTOKEN_500": int(get_parameter("rcs-fond-premium-user-maxtoken-500")),
            "MAXTOKEN_300": int(get_parameter("rcs-fond-premium-user-maxtoken-300")),
            "MAXTOKEN_150": int(get_parameter("rcs-fond-premium-user-maxtoken-150")),
            "MODELID_EMBED": get_parameter("rcs-fond-premium-user-modelId-embed"),
            "GUARDRAIL_IDENTIFIER": get_parameter("rcs-fond-premium-user-guardrail-identifier"),
            "GUARDRAIL_VERSION": get_parameter("rcs-fond-premium-user-guardrail-version"),
            "PROMPT_BUCKET": get_parameter("rcs-fond-premium-user-s3-bucket-prompt"),
            "PROMPT_PREFIX": get_parameter("rcs-fond-premium-user-s3-bucket-prompt-prefix")
        }
        logger.info("Successfully loaded configuration from Parameter Store")
    except Exception as e:
        logger.error(f"Failed to load configuration from Parameter Store: {str(e)}")
        raise

def load_prompt_from_s3(prompt_filename):
    """Load prompt from S3 bucket"""
    try:
        prompt_key = f"{config['PROMPT_PREFIX']}/{prompt_filename}"
        logger.info(f"Loading prompt from S3: {prompt_key}")
        
        response = s3_client.get_object(
            Bucket=config['PROMPT_BUCKET'],
            Key=prompt_key
        )
        
        prompt_content = response['Body'].read().decode('utf-8')
        logger.info(f"Successfully loaded prompt: {prompt_filename}")
        return prompt_content
    except Exception as e:
        logger.error(f"Error loading prompt {prompt_filename}: {str(e)}")
        raise

# Global variable to store all prompts
prompts = {}

def load_all_prompts():
    """Load all required prompts from S3"""
    try:
        logger.info("Loading all prompts from S3")
        global prompts
        prompts = {
            "SYSTEM_PROMPT": load_prompt_from_s3("system prompt.txt"),
            "SQL_SYSTEM_PROMPT": load_prompt_from_s3("sql system prompt.txt"),
            "GET_SQL_QUERY_PROMPT": load_prompt_from_s3("get sql query system prompt.txt"),
            "FORMAT_ATHENA_RESULTS_PROMPT": load_prompt_from_s3("formate athena results system prompt.txt"),
            "PROCESS_SEARCH_RESULTS_PROMPT": load_prompt_from_s3("process search result system prompt..txt")
        }
        logger.info("Successfully loaded all prompts from S3")
    except Exception as e:
        logger.error(f"Failed to load prompts from S3: {str(e)}")
        raise

# Tool definitions - moved to a function to use config values
def get_tools():
    return [
        {
            "toolSpec": {
                "name": "retrieve_chat_history",
                "description": "Retrieve and display previous conversation history",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query about retrieving chat history"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "generate_sql",
                "description": "Generate SQL query for pet data queries",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Natural language question about pet data"
                            }
                        },
                        "required": ["question"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "search_documents",
                "description": "Search through document content",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query text"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        }
    ]

def get_sql_query(question, actual_pet_name=None):
    """Generate SQL query with enhanced error handling and validation"""
    try:
        # Determine activity type
        activity_types = {
            'walk': 'walk',
            'play': 'play',
            'rest': 'rest',
            'sleep': 'sleep',
            'activity': '*',
            'latest': '*'
        }

        # Extract name from question
        name_match = re.search(r'([A-Za-z]+)\'s', question)
        mentioned_name = name_match.group(1) if name_match else None
        
        # Use the actual pet name from database if available, otherwise use mentioned name
        name_to_use = actual_pet_name or mentioned_name
        
        # Check for year
        year_match = re.search(r'(\d{4})', question)
        year = year_match.group(1) if year_match else None
        
        activity_type = next((activity_types[key] for key in activity_types if key in question.lower()), 'sleep')
        
        # Create a more focused prompt for SQL generation that uses the correct pet name
        system_prompt = [{"text": prompts["GET_SQL_QUERY_PROMPT"].format(name_to_use=name_to_use)}]

        max_retries = 3
        retry_count = 0
        while True:
            try:
                response = bedrock_client.converse(
                    modelId=config["MODEL_ID"],
                    system=system_prompt,
                    messages=[{
                        "role": "user",
                        "content": [{"text": f"Generate SQL query for: {question}"}]
                    }],
                    inferenceConfig={
                        "temperature": 0.1,
                        "maxTokens": config["MAXTOKEN_500"]
                    },
                    guardrailConfig={
                        "guardrailIdentifier": config["GUARDRAIL_IDENTIFIER"],
                        "guardrailVersion": config["GUARDRAIL_VERSION"]
                    }
                )
                break
            except Exception as e:
                if "ThrottlingException" in str(e) and retry_count < max_retries:
                    retry_count += 1
                    wait_time = (2 ** retry_count - 1)
                    time.sleep(wait_time)
                    continue
                raise

        # Enhanced response processing
        if 'output' in response and 'message' in response['output']:
            message = response['output']['message']
            for item in message['content']:
                if isinstance(item, dict) and 'text' in item:
                    # Clean up the SQL
                    sql = item['text'].strip()
                    sql = re.sub(r'```sql|```', '', sql).strip()
                    
                    # Validate SQL structure
                    if sql.upper().startswith('SELECT') and 'FROM' in sql.upper():
                        # Add semicolon if missing
                        if not sql.rstrip().endswith(';'):
                            sql += ';'
                            
                        logger.info(f"Generated valid SQL query: {sql}")
                        return sql, activity_type
                    else:
                        logger.warning(f"Invalid SQL structure generated: {sql}")
                        
        logger.warning("No valid SQL generated from Bedrock response")
        return None, activity_type
        
    except Exception as e:
        logger.error(f"SQL generation error: {str(e)}")
        return None, activity_type

def execute_athena_query(query):
    """Execute Athena query and format results"""
    try:
        logger.info(f"Starting Athena query execution: {query}")
        query_start_time = time.time()
        
        response = athena_client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': config["ATHENA_DATABASE"]},
            ResultConfiguration={'OutputLocation': config["ATHENA_OUTPUT_BUCKET"]}
        )
        
        query_execution_id = response['QueryExecutionId']
        logger.info(f"Athena query submitted - Execution ID: {query_execution_id}")
        
        poll_count = 0
        while time.time() - query_start_time < 20:
            poll_count += 1
            status = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
            state = status['QueryExecution']['Status']['State']
            
            if state == 'SUCCEEDED':
                results = athena_client.get_query_results(QueryExecutionId=query_execution_id)
                execution_time = time.time() - query_start_time
                logger.info(f"Athena query succeeded - Time: {execution_time:.2f}s, Polls: {poll_count}")
                
                if 'ResultSet' not in results or 'Rows' not in results['ResultSet']:
                    logger.info("Athena query returned empty result set")
                    return []
                    
                row_count = len(results['ResultSet']['Rows']) - 1
                logger.info(f"Athena query returned {row_count} rows")
                
                headers = [col['Label'] for col in results['ResultSet']['ResultSetMetadata']['ColumnInfo']]
                return [{headers[i]: value.get('VarCharValue', '') 
                        for i, value in enumerate(row['Data'])}
                       for row in results['ResultSet']['Rows'][1:]]
                       
            elif state in ['FAILED', 'CANCELLED']:
                error_reason = status['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                logger.error(f"Athena query failed - ID: {query_execution_id}, Error: {error_reason}")
                return []
                
            time.sleep(0.3)
            
        logger.error(f"Athena query timeout after {time.time() - query_start_time:.2f}s - ID: {query_execution_id}")
        return []
            
    except Exception as e:
        logger.error(f"Athena API error: {str(e)}")
        return []

def format_athena_results(results, activity_type='sleep'):
    """Generate friendly analysis of pet activity data"""
    if not results:
        return "I don't see any activity data for that time period."
    
    try:
        system_prompt = prompts["FORMAT_ATHENA_RESULTS_PROMPT"]
        
        max_retries = 3
        retry_count = 0
        while True:
            try:
                response = bedrock_client.converse(
                    modelId=config["MODEL_ID"],
                    system=[{"text": system_prompt}],
                    messages=[{
                        "role": "user",
                        "content": [{"text": f"Analyze this {activity_type} data and provide friendly feedback: {json.dumps(results)}"}]
                    }],
                    inferenceConfig={
                        "temperature": config["TEMPERATURE"],
                        "maxTokens": config["MAXTOKEN_300"]
                    },
                    guardrailConfig={
                        "guardrailIdentifier": config["GUARDRAIL_IDENTIFIER"],
                        "guardrailVersion": config["GUARDRAIL_VERSION"]
                    }
                )
                break
            except Exception as e:
                if "ThrottlingException" in str(e) and retry_count < max_retries:
                    retry_count += 1
                    wait_time = (2 ** retry_count - 1)
                    time.sleep(wait_time)
                    continue
                raise
        
        return response['output']['message']['content'][0]['text'].strip()
        
    except Exception as e:
        logger.error(f"Analysis formatting error: {str(e)}")
        return f"I'm having trouble analyzing the {activity_type} data right now."

def process_search_results(query, results):
    """Process and format search results"""
    try:
        system_prompt = prompts["PROCESS_SEARCH_RESULTS_PROMPT"]
        
        max_retries = 3
        retry_count = 0
        while True:
            try:
                response = bedrock_client.converse(
                    modelId=config["MODEL_ID"],
                    system=[{"text": system_prompt}],
                    messages=[{
                        "role": "user", 
                        "content": [{"text": f"Help with this pet care question: {query}\nUsing this information: {json.dumps(results)}"}]
                    }],
                    inferenceConfig={
                        "temperature": config["TEMPERATURE"],
                        "maxTokens": config["MAXTOKEN_500"]
                    },
                    guardrailConfig={
                        "guardrailIdentifier": config["GUARDRAIL_IDENTIFIER"],
                        "guardrailVersion": config["GUARDRAIL_VERSION"]
                    }
                )
                break
            except Exception as e:
                if "ThrottlingException" in str(e) and retry_count < max_retries:
                    retry_count += 1
                    wait_time = (2 ** retry_count - 1)
                    time.sleep(wait_time)
                    continue
                raise
        
        return {
            "response": response['output']['message']['content'][0]['text'].strip(),
            "sources": list(set(r['file'].replace('pdf-uploads/', '').replace('-compressed.pdf', '') 
                              for r in results))[:5]
        }
    except Exception as e:
        logger.error(f"Response formatting error: {str(e)}")
        return {"response": "", "sources": []}

def search_documents(query):
    """Combined document search functionality"""
    try:
        logger.info(f"Starting OpenSearch query: {query}")
        search_start_time = time.time()
        
        credentials = boto3.Session().get_credentials()
        opensearch_client = OpenSearch(
            hosts=[{'host': config["OPENSEARCH_ENDPOINT"].replace('https://', ''), 'port': 443}],
            http_auth=AWSV4SignerAuth(credentials, config["OPENSEARCH_REGION"], 'aoss'),
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            pool_maxsize=20
        )

        # Generate embeddings
        logger.info("Calling Bedrock for embedding generation")
        embedding_start_time = time.time()
        
        body = json.dumps({
            "texts": [query],
            "input_type": "search_query",
            "embedding_types": ["float"]
        })
        max_retries = 3
        retry_count = 0
        while True:
            try:
                response = bedrock_client.invoke_model(
                    body=body,
                    modelId=config["MODELID_EMBED"],
                    accept='*/*',
                    contentType='application/json'
                )
                break
            except Exception as e:
                if "ThrottlingException" in str(e) and retry_count < max_retries:
                    retry_count += 1
                    wait_time = (2 ** retry_count - 1)
                    time.sleep(wait_time)
                    continue
                raise
        
        logger.info(f"Embedding generated in {time.time() - embedding_start_time:.2f}s")
        embedding = json.loads(response['body'].read())['embeddings']['float'][0]

        # Search OpenSearch
        logger.info("Executing OpenSearch query")
        results = opensearch_client.search(
            index=config["INDEX_NAME"],
            body={
                "size": 5,
                "query": {
                    "knn": {
                        "vector_field": {
                            "vector": embedding,
                            "k": 5,
                            "filter": {"match_all": {}}
                        }
                    }
                },
                "_source": ["content", "file_name"]
            }
        )

        total_time = time.time() - search_start_time
        hit_count = len(results['hits']['hits'])
        logger.info(f"OpenSearch query completed - Time: {total_time:.2f}s, Hits: {hit_count}")

        formatted_results = [
            {
                'content': hit['_source'].get('content', ''),
                'file': hit['_source'].get('file_name', ''),
                'relevance': hit.get('_score', 0)
            }
            for hit in results['hits']['hits']
        ]

        return [r for r in formatted_results if r['relevance'] > 0.5]

    except Exception as e:
        logger.error(f"OpenSearch error: {str(e)}")
        return []

def store_chat_history(session_id, user_id, pet_id, query, response, user_type="premium user"):
    """Store chat interaction in DynamoDB with accumulated history"""
    try:
        timestamp = datetime.utcnow().isoformat()
        chat_table = dynamodb.Table(config["CHATHISTORY_TABLE"])
        
        # First, try to get existing item
        try:
            existing_item = chat_table.get_item(
                Key={
                    'session_id': session_id,
                    'user_type': user_type
                }
            ).get('Item', {})
        except:
            existing_item = {}

        # Initialize or get existing chat history
        chat_history = existing_item.get('chat_history', {'conversations': []})
        
        # Add new conversation to history
        chat_history['conversations'].append({
            'query': query,
            'response': response,
            'timestamp': timestamp
        })

        # Update or create item in DynamoDB
        chat_table.put_item(
            Item={
                'session_id': session_id,
                'user_type': user_type,
                'user_id': str(user_id),
                'pet_id': str(pet_id),
                'chat_history': chat_history,
                'timestamp': timestamp
            }
        )
    except Exception as e:
        logger.error(f"Failed to store chat history: {str(e)}")

def get_recent_chat_history(session_id, user_type="premium user"):
    """Retrieve chat history for context"""
    try:
        chat_table = dynamodb.Table(config["CHATHISTORY_TABLE"])
        response = chat_table.get_item(
            Key={
                'session_id': session_id,
                'user_type': user_type
            }
        )
        
        if 'Item' in response:
            chat_history = response['Item'].get('chat_history', {}).get('conversations', [])
            # Sort by timestamp if needed
            chat_history.sort(key=lambda x: x['timestamp'])
            return chat_history
        return []
    except Exception as e:
        logger.error(f"Failed to retrieve chat history: {str(e)}")
        return []

def generate_response(context, query, chat_history=None):
    """Improved generate_response with better greeting detection and tool handling"""
    try:
        # Parse context to extract pet name
        context_data = json.loads(context)
        actual_pet_name = context_data.get('petName', '')

        # Count the number of questions in a query
        question_count = len([q.strip() for q in query.split('?') if q.strip()])
        if question_count > 2:
            return {
                "response": "I notice you have several important questions about your pet's health. To make sure I can give you the most thorough and helpful answer for each question, could you please ask 1-2 questions at a time? This way, we can address each of your concerns with the attention it deserves.",
                "sources": []
            }

        # Format chat history for context
        history_context = ""
        if chat_history:
            history_context = "\nPrevious conversation:\n" + "\n".join(
                f"User: {conv['query']}\nAssistant: {conv['response']}"
                for conv in chat_history[-3:] 
            )

        # Check for greetings or casual conversation first
        query_lower = query.lower().strip()
        greetings = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 
            'howdy', 'hiya', 'greetings', 'what\'s up', 'how are you', 'how\'s it going',
            'what can you help me with?', 'what can you do for me?', 'thank you'
        ]
        is_greeting = any(query_lower == greeting or query_lower.startswith(greeting + ' ') for greeting in greetings)
        
        if is_greeting:
            logger.info(f"Detected greeting or casual conversation: {query}")
            
            # For greetings, use direct generation without tools
            max_retries = 3
            retry_count = 0
            while True:
                try:
                    greeting_response = bedrock_client.converse(
                        modelId=config["MODEL_ID"],
                        system=[{"text": "You are a friendly pet care assistant. Respond warmly to greetings and casual conversation, keeping responses brief and natural. Don't use formal language or mention tools/searching."}],
                        messages=[
                            {
                                "role": "user", 
                                "content": [{"text": f"{query}{history_context}"}]
                            }
                        ],
                        inferenceConfig={
                            "temperature": 0.3,
                            "maxTokens": config["MAXTOKEN_150"]
                        },
                        guardrailConfig={
                            "guardrailIdentifier": config["GUARDRAIL_IDENTIFIER"],
                            "guardrailVersion": config["GUARDRAIL_VERSION"]
                        }
                    )
                    break
                except Exception as e:
                    if "ThrottlingException" in str(e) and retry_count < max_retries:
                        retry_count += 1
                        wait_time = (2 ** retry_count - 1)
                        time.sleep(wait_time)
                        continue
                    raise
            
            return {
                "response": greeting_response['output']['message']['content'][0]['text'].strip(),
                "sources": []
            }
        
        # Check if this is explicitly about previous conversation
        is_history_request = any(term in query.lower() for term in [
            'previous conversation', 'last conversation', 'chat history', 
            'previous chat', 'last time', 'previous message', 'earlier message',
            'previous question', 'last question', 'earlier question', 
            'what did i ask', 'what did we discuss', 'what was my last', 
            'what was my previous'  
        ])

        if is_history_request:
            logger.info(f"Detected history request: '{query}' - Directing to history retrieval tool")
        
        # Extract pet name to check for activity analysis
        name_match = re.search(r'([A-Za-z]+)\'s', query) or re.search(r'\b(charlie|max|bella|fido|rex|spot|buddy|luna|daisy)\b', query.lower())
        pet_name_mentioned = bool(name_match)
        activity_terms = ['sleep', 'rest', 'walk', 'play', 'activity', 'getting', 'doing']
        is_activity_request = pet_name_mentioned and any(term in query.lower() for term in activity_terms)
        
        # Set appropriate tool choice based on query type
        tool_choice = {"auto": {}}  # Default
        
        if is_history_request:
            tool_choice = {"tool": {"name": "retrieve_chat_history"}}
            logger.info("Directing to history retrieval tool")
        elif is_activity_request:
            tool_choice = {"tool": {"name": "generate_sql"}}
            logger.info("Directing to SQL generation tool")
        else:
            # For general questions, use search_documents
            tool_choice = {"tool": {"name": "search_documents"}}
            logger.info("Directing to document search tool")

        # Make the API call with appropriate tool choice
        response = bedrock_client.converse(
            modelId=config["MODEL_ID"],
            system=[{"text": prompts["SYSTEM_PROMPT"]}],
            messages=[
                {
                    "role": "user", 
                    "content": [{"text": f"Context: {context}\nQuery: {query}{history_context}"}]
                }
            ],
            inferenceConfig={
                "temperature": config["TEMPERATURE"],
                "maxTokens": config["MAXTOKEN_500"]
            },
            toolConfig={
                "tools": get_tools(), 
                "toolChoice": tool_choice
            },
            guardrailConfig={
                "guardrailIdentifier": config["GUARDRAIL_IDENTIFIER"],
                "guardrailVersion": config["GUARDRAIL_VERSION"]
            }
        )

        final_response = ""
        sources = []

        # Process the response
        for item in response['output']['message']['content']:
            if isinstance(item, dict) and 'toolUse' in item:
                tool_use = item['toolUse']
                
                if tool_use['name'] == 'retrieve_chat_history':
                    # --- INLINE FUNCTION: retrieve_chat_history_tool ---
                    # Session ID is part of context data
                    context_data = json.loads(context)
                    session_id = context_data.get('sessionId', '')
                    
                    # Get chat history directly
                    chat_history = get_recent_chat_history(session_id, user_type="premium user")
                    
                    if not chat_history:
                        final_response = "I couldn't find any recent conversation history."
                    else:
                        # Check if the query is asking for just the previous question
                        is_previous_question_request = any(term in query.lower() for term in [
                            'previous question', 'last question', 'what was my last', 
                            'what was my previous question', 'what did i ask last'
                        ])
                        
                        if is_previous_question_request and len(chat_history) >= 1:
                            # Return only the most recent user question
                            final_response = f"Your previous question was: '{chat_history[-1]['query']}'"
                        else:
                            # Format the conversation history
                            final_response = "Here's a summary of our recent conversation:\n\n"
                            for interaction in chat_history[-5:]:  
                                final_response += f"🗨️ User: {interaction['query']}\n"
                                final_response += f"🤖 Assistant: {interaction['response']}\n\n"

                elif tool_use['name'] == 'generate_sql':
                    sql_query, activity_type = get_sql_query(tool_use['input']['question'], actual_pet_name)
                    if sql_query:
                        results = execute_athena_query(sql_query)
                        if results:
                            final_response = format_athena_results(results, activity_type)
                        else:
                            final_response = f"I don't see any activity data for that time period."
                
                elif tool_use['name'] == 'search_documents':
                    search_results = search_documents(tool_use['input']['query'])
                    if search_results:
                        doc_results = process_search_results(tool_use['input']['query'], search_results)
                        final_response = doc_results['response']
                        sources = doc_results['sources']
                    else:
                        final_response = "I don't have specific information about that, but I'd be happy to help with other pet care questions."
            
            elif isinstance(item, dict) and 'text' in item:
                final_response += item['text']

        # If we still don't have a response, provide a friendly fallback
        if not final_response.strip():
            final_response = "I understand your question about pet care. While I don't have specific information on that, I'd be happy to help with other pet-related questions you might have."

        return {
            "response": final_response.strip(),
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Response generation error: {str(e)}")
        return {
            "response": "I encountered an unexpected error while processing your request. Please try again.",
            "sources": []
        }

def lambda_handler(event, context):
    """Updated handler with chat history integration, improved context passing, and pet name validation"""
    try:
        start_time = time.time()
        logger.info(f"Lambda invocation started - Request ID: {context.aws_request_id}")
        
        # Load configuration from Parameter Store and prompts from S3
        load_config_from_parameter_store()
        load_all_prompts()
        
        # Extract and validate input fields
        user_id = event.get('UserId', '').strip()
        pet_id = event.get('PetId', '').strip()
        query = event.get('Querry', '').strip()
        session_id = event.get('SessionId', '').strip()
        
        if not all([query, user_id, pet_id, session_id]):
            logger.error("Missing required fields")
            return {
                'statusCode': 400, 
                'body': json.dumps({'error': 'UserId, PetId, SessionId, and Querry fields are required'})
            }
            
        try:
            user_id = int(user_id)
            pet_id = int(pet_id)
        except ValueError:
            logger.error(f"Invalid ID format - UserId: {user_id}, PetId: {pet_id}")
            return {
                'statusCode': 400, 
                'body': json.dumps({'error': 'UserId and PetId must be valid numbers'})
            }
        
        # Make sure config is loaded before accessing it
        if not config or "ATHENA_TABLES" not in config:
            logger.error("Configuration not properly loaded")
            return {
                'statusCode': 500, 
                'body': json.dumps({'error': 'Service configuration error'})
            }
            
        # Get the actual pet name from database for verification
        pet_name_query = f"""
        SELECT DISTINCT name
        FROM {config["ATHENA_TABLES"]["pet_record"]}
        WHERE userid = {user_id}
        AND petid = {pet_id}
        LIMIT 1;
        """
        
        pet_name_results = execute_athena_query(pet_name_query)
        
        # if not pet_name_results:
        #     logger.error(f"User-pet validation failed - UserId: {user_id}, PetId: {pet_id}")
        #     return {
        #         'statusCode': 403, 
        #         'body': json.dumps({'error': 'Invalid UserId or PetId combination'})
        #     }
        if not pet_name_results:
            logger.error(f"User-pet validation failed - UserId: {user_id}, PetId: {pet_id}")
            return {
                'statusCode': 200,  
                'body': json.dumps({
                    "response": "I couldn't find information for this pet and user combination. Please check that you're using the correct user ID and pet ID.",
                    "sources": []
                })
            }
        actual_pet_name = pet_name_results[0].get('name', '').lower()
        
        # Extract mentioned pet name from query to validate
        name_match = re.search(r'([A-Za-z]+)\'s', query) or re.search(r'\b(charlie|max|bella|fido|rex|spot|buddy|luna|daisy)\b', query.lower())
        
        if name_match and name_match.group(1).lower() != actual_pet_name:
            logger.warning(f"Pet name mismatch - Queried: {name_match.group(1)}, Actual: {actual_pet_name}")
            response_message = f"I notice you're asking about {name_match.group(1)}, but the pet associated with this profile is {actual_pet_name.title()}. I can only provide information about {actual_pet_name.title()} with this pet ID. Would you like to ask something about {actual_pet_name.title()} instead?"
            
            # Store the interaction
            store_chat_history(session_id, user_id, pet_id, query, response_message)
            
            return {
                'statusCode': 200, 
                'body': json.dumps({
                    "response": response_message,
                    "sources": []
                })
            }
        
        # Retrieve recent chat history
        chat_history = get_recent_chat_history(session_id)
        
        # Generate response with context and chat history
        context_data = {
            'tables': config["ATHENA_TABLES"],
            'petId': pet_id,
            'userId': user_id,
            'sessionId': session_id,
            'petName': actual_pet_name  
        }
        
        result = generate_response(json.dumps(context_data), query, chat_history)
        
        # Store the new interaction
        store_chat_history(session_id, user_id, pet_id, query, result['response'])
        
        total_time = time.time() - start_time
        logger.info(f"Lambda execution completed - Total time: {total_time:.2f}s")
        
        return {'statusCode': 200, 'body': json.dumps(result)}
        
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}