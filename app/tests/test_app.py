from unittest import TestCase
from functions import get_number


class TestApp(TestCase):
    
    def test_get_number(self):
        result = get_number(number=10)
        self.assertEqual(result, 10)
    