#!/bin/bash
# Test script for the new function-based pipeline

echo "========================================"
echo "Testing Function Pipeline Implementation"
echo "========================================"

# Test data (wrapped in content field as expected by API)
BLOG_POST='{
  "content": {
    "id": "test_blog_001",
    "title": "10 Tips for Effective Marketing Automation",
    "content": "Marketing automation has become essential for modern businesses. Here are 10 proven strategies to maximize your marketing automation efforts.\n\n1. Define Clear Goals: Start with specific, measurable objectives.\n2. Segment Your Audience: Tailor messages to specific customer groups.\n3. Create Quality Content: Focus on value and relevance.\n4. Use Multi-Channel Approach: Integrate email, social media, and web.\n5. Monitor and Optimize: Continuously track performance metrics.\n6. Personalize Communications: Use customer data effectively.\n7. Test Everything: A/B test subject lines, content, and CTAs.\n8. Maintain Data Quality: Keep your database clean and updated.\n9. Align Sales and Marketing: Ensure teams work together.\n10. Stay Compliant: Follow GDPR and other regulations.\n\nImplementing these strategies will help you build more effective marketing campaigns and drive better ROI.",
    "snippet": "Marketing automation essentials: 10 proven strategies to maximize your automation efforts and drive better results.",
    "author": "Marketing Team",
    "category": "Marketing Strategy",
    "tags": ["marketing", "automation", "strategy", "best practices"],
    "created_at": "2025-10-27T10:00:00Z"
  }
}'

echo ""
echo "1. Testing blog post processor with function pipeline..."
echo "   Endpoint: POST /api/v1/process/blog"
echo ""

curl -X POST "http://localhost:8000/api/v1/process/blog" \
  -H "Content-Type: application/json" \
  -d "$BLOG_POST" \
  -o /tmp/pipeline_result.json \
  -w "\nHTTP Status: %{http_code}\n"

echo ""
echo "2. Checking job status..."
JOB_ID=$(cat /tmp/pipeline_result.json | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)
echo "   Job ID: $JOB_ID"

if [ -n "$JOB_ID" ]; then
  echo ""
  echo "3. Waiting for job to complete (30 seconds)..."
  sleep 30

  echo ""
  echo "4. Fetching job result..."
  curl -X GET "http://localhost:8000/api/v1/jobs/$JOB_ID/result" \
    -o /tmp/job_result.json \
    -w "\nHTTP Status: %{http_code}\n"

  echo ""
  echo "5. Result summary:"
  cat /tmp/job_result.json | python3 -m json.tool | head -50

  echo ""
  echo "6. Checking pipeline_result structure:"
  cat /tmp/job_result.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'result' in data and 'pipeline_result' in data['result']:
    pr = data['result']['pipeline_result']
    print(f'  Pipeline Status: {pr.get(\"pipeline_status\", \"unknown\")}')
    print(f'  Steps Completed: {len(pr.get(\"step_results\", {}))}')
    if 'step_results' in pr:
        print(f'  Step Names: {list(pr[\"step_results\"].keys())}')
    print(f'  Has Final Content: {\"final_content\" in pr and len(pr.get(\"final_content\", \"\")) > 0}')
    if 'metadata' in pr:
        print(f'  Execution Time: {pr[\"metadata\"].get(\"execution_time_seconds\", \"N/A\")} seconds')
        print(f'  Total Tokens: {pr[\"metadata\"].get(\"total_tokens_used\", \"N/A\")}')
else:
    print('  ERROR: pipeline_result not found in expected structure')
"
else
  echo "   ERROR: Could not extract job_id from response"
  echo ""
  echo "   Response content:"
  cat /tmp/pipeline_result.json | python3 -m json.tool
fi

echo ""
echo "========================================"
echo "Test Complete"
echo "========================================"
