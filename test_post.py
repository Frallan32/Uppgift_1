import  unittest #import TestCase,unittest
from post import Post

class PostTest(unittest.TestCase):
    def test_create_post(self):
        #1-Instancing from source code
        #2- adapt instance till testing function
        #3- assert (validate expectations Vs. results) 
        #1- instancing : 
        p =Post(title='Test',
                content='My 1st test')
        #2- No adaptation needed as no return
        #3- Assert or validation
        ## My Expectations are : 
        # ### A : p will have title = 'Test'
        # ### B : p will have content ='My 1st test'
        # ### A:
        self.assertEqual('Test',p.title)
        # ### B:
        self.assertEqual('My 1st test',p.content)
    
    def test_export_to_json(self):
        #1-Instancing from source code
        #2- adapt instance till testing function
        #3- assert (validate expectations Vs. results) 
        #1- Instancing:
        p = Post(title='Test2',
                 content="Test2 will be nice")
        #2- Expectations : 
        expected = {'title':'Test2','content':"Test2 will be nice"}
        #3- Assert or Validate
        self.assertDictEqual(expected,\
                             p.export_to_json())

if __name__ == '__main__':
    unittest.main()