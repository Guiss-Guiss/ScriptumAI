import unittest
from query_component import QueryComponent

class TestQueryComponent(unittest.TestCase):
    def setUp(self):
        self.query_component = QueryComponent()

    def test_process_query_basic(self):
        query = "What is the capital of France?"
        result = self.query_component.process_query(query)
        
        self.assertEqual(result['original_query'], query)
        self.assertEqual(result['processed_query'], "what is the capital of france?")
        self.assertIsInstance(result['options'], dict)

    def test_process_query_with_options(self):
        query = "List all employees"
        options = {"limit": 10, "department": "IT"}
        result = self.query_component.process_query(query, options)
        
        self.assertEqual(result['original_query'], query)
        self.assertEqual(result['processed_query'], "list all employees")
        self.assertEqual(result['options'], options)

    def test_process_query_with_whitespace(self):
        query = "  How does photosynthesis work?  "
        result = self.query_component.process_query(query)
        
        self.assertEqual(result['original_query'], query)
        self.assertEqual(result['processed_query'], "how does photosynthesis work?")

    def test_process_query_empty(self):
        query = ""
        result = self.query_component.process_query(query)
        
        self.assertEqual(result['original_query'], "")
        self.assertEqual(result['processed_query'], "")

if __name__ == '__main__':
    unittest.main()