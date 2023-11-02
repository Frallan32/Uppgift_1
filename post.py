#post.py

class Post:
    def __init__(self,title:str,content:str) -> None:
        self.title = str(title)
        self.content = content

    def export_to_json(self):
        return self.__dict__
    
    def __str__(self) -> str:
        return f"Post_title = {self.title} "+\
        f"Post_content = {self.content}"


if __name__ =='__main__':
    my_post = Post("Python","Python is Great")
    print(my_post.export_to_json())