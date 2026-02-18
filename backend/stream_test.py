import sys
from medical_chatbot_simple import run_medical_rag_stream, PDF_PATH

# Test streaming functionality
print("🏥 Testing Streaming Medical Chatbot")
print("=" * 40)
print("\nQuestion: What are the symptoms of diabetes?")
print("\nResponse (streaming):")
print("-" * 30)

try:
    answer = run_medical_rag_stream(PDF_PATH, "What are the symptoms of diabetes?")
    print(f"\n\nFull response length: {len(answer)} characters")
    print("✅ Streaming test completed successfully!")
except Exception as e:
    print(f"❌ Error: {str(e)}")