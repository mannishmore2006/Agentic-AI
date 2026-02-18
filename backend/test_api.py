import requests
import json

# Test the medical chatbot API
def test_medical_chatbot():
    url = "http://localhost:8000/ask"
    
    # Test questions
    test_questions = [
        "What is diabetes?",
        "What are common symptoms of heart disease?",
        "How to prevent infections?"
    ]
    
    print("🏥 Testing Medical Chatbot API")
    print("=" * 40)
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        print("-" * 30)
        
        try:
            response = requests.post(url, 
                                   json={"question": question},
                                   timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Answer: {result['answer'][:200]}...")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {str(e)}")
    
    # Test health endpoint
    print("\n" + "=" * 40)
    print("🏥 Checking API Health...")
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=30)
        if health_response.status_code == 200:
            print("✅ API is healthy and running!")
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")

if __name__ == "__main__":
    test_medical_chatbot()