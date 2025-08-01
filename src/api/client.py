
import requests
import json
import time
from typing import List, Dict, Any

# API configuration
API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("🏥 Testing health endpoint...")
    response = requests.get(f"{API_BASE_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health check passed: {data}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def test_model_info():
    """Test the model info endpoint."""
    print("\n📊 Testing model info endpoint...")
    response = requests.get(f"{API_BASE_URL}/model-info")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Model info: {json.dumps(data, indent=2)}")
        return True
    else:
        print(f"❌ Model info failed: {response.status_code}")
        return False

def test_single_classification():
    """Test single product classification."""
    print("\n🔍 Testing single classification...")
    
    # Test data
    test_text = "Samsung Galaxy S21 Smartphone avec 128GB de stockage, écran 6.2 pouces, appareil photo 64MP"
    
    payload = {
        "text": test_text,
        "use_cache": True
    }
    
    response = requests.post(f"{API_BASE_URL}/classify", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Classification result:")
        print(f"   Text: {data['text'][:50]}...")
        print(f"   Predicted: {data['predicted_label']}")
        print(f"   Confidence: {data.get('confidence', 'N/A')}")
        print(f"   Processing time: {data['processing_time_ms']:.2f}ms")
        print(f"   Cached: {data['cached']}")
        return True
    else:
        print(f"❌ Classification failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_batch_classification():
    """Test batch classification."""
    print("\n📦 Testing batch classification...")
    
    # Test data
    test_texts = [
        "Sony WH-1000XM4 Casque sans fil avec réduction de bruit, Bluetooth, 30h d'autonomie",
        "Apple MacBook Pro 13 pouces, processeur M1, 8GB RAM, 256GB SSD, macOS",
        "Nike Air Max 270 Chaussures de sport, amorti Air Max, semelle en caoutchouc",
        "Dell XPS 15 Ordinateur portable 15.6 pouces, Intel i7, 16GB RAM, 512GB SSD",
        "Canon EOS R6 Appareil photo hybride, capteur plein format, 4K, stabilisation"
    ]
    
    payload = {
        "texts": test_texts,
        "use_cache": True,
        "limit": 3  # Test with limit
    }
    
    response = requests.post(f"{API_BASE_URL}/classify/batch", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Batch classification result:")
        print(f"   Total processing time: {data['total_processing_time_ms']:.2f}ms")
        print(f"   Average processing time: {data['average_processing_time_ms']:.2f}ms")
        print(f"   Number of predictions: {len(data['predictions'])} (limited to 3)")
        
        for i, pred in enumerate(data['predictions']):
            print(f"   {i+1}. {pred['text'][:40]}... → {pred['predicted_label']} (cached: {pred['cached']})")
        
        return True
    else:
        print(f"❌ Batch classification failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_examples():
    """Test the examples endpoint."""
    print("\n📝 Testing examples endpoint...")
    response = requests.get(f"{API_BASE_URL}/examples")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Examples retrieved: {data['count']} examples")
        for i, example in enumerate(data['examples']):
            print(f"   {i+1}. {example[:60]}...")
        return True
    else:
        print(f"❌ Examples failed: {response.status_code}")
        return False

def test_cache_stats():
    """Test cache statistics."""
    print("\n💾 Testing cache stats...")
    response = requests.get(f"{API_BASE_URL}/cache/stats")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Cache stats: {json.dumps(data, indent=2)}")
        return True
    else:
        print(f"❌ Cache stats failed: {response.status_code}")
        return False

def test_clear_cache():
    """Test cache clearing."""
    print("\n🗑️ Testing cache clearing...")
    response = requests.delete(f"{API_BASE_URL}/cache")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Cache cleared: {data['message']}")
        return True
    else:
        print(f"❌ Cache clearing failed: {response.status_code}")
        return False

def test_limited_classification():
    """Test limited classification endpoint."""
    print("\n🎯 Testing limited classification...")
    
    # Test data
    test_texts = [
        "Sony WH-1000XM4 Casque sans fil avec réduction de bruit, Bluetooth, 30h d'autonomie",
        "Apple MacBook Pro 13 pouces, processeur M1, 8GB RAM, 256GB SSD, macOS",
        "Nike Air Max 270 Chaussures de sport, amorti Air Max, semelle en caoutchouc",
        "Dell XPS 15 Ordinateur portable 15.6 pouces, Intel i7, 16GB RAM, 512GB SSD",
        "Canon EOS R6 Appareil photo hybride, capteur plein format, 4K, stabilisation"
    ]
    
    payload = {
        "texts": test_texts,
        "limit": 2,
        "use_cache": True
    }
    
    response = requests.post(f"{API_BASE_URL}/classify/limited", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Limited classification result:")
        print(f"   Processed {len(data)} texts (limited from {len(test_texts)})")
        
        for i, pred in enumerate(data):
            print(f"   {i+1}. {pred['text'][:40]}... → {pred['predicted_label']}")
        
        return True
    else:
        print(f"❌ Limited classification failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def interactive_classification():
    """Interactive classification mode."""
    print("\n🎯 Interactive classification mode")
    print("Enter product descriptions (or 'quit' to exit):")
    
    while True:
        text = input("\nProduct description: ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            continue
        
        payload = {
            "text": text,
            "use_cache": True
        }
        
        try:
            response = requests.post(f"{API_BASE_URL}/classify", json=payload)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Prediction: {data['predicted_label']}")
                if data.get('confidence'):
                    print(f"   Confidence: {data['confidence']:.3f}")
                print(f"   Time: {data['processing_time_ms']:.2f}ms")
                print(f"   Cached: {data['cached']}")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Request failed: {e}")

def main():
    """Run all tests."""
    print("🚀 Starting Mistral Product Classification API tests...")
    print(f"📍 API URL: {API_BASE_URL}")
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("❌ API is not responding. Make sure the server is running:")
            print("   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running:")
        print("   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
        return
    
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Single Classification", test_single_classification),
        ("Batch Classification", test_batch_classification),
        ("Limited Classification", test_limited_classification),
        ("Examples", test_examples),
        ("Cache Stats", test_cache_stats),
        ("Clear Cache", test_clear_cache),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
        print("\n💡 Try interactive mode:")
        interactive_classification()
    else:
        print("⚠️ Some tests failed. Check the API configuration.")

if __name__ == "__main__":
    main() 