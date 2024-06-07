from transformers import CLIPProcessor, CLIPModel

# "openai/clip-vit-large-patch14"
class ClipWrapper:

    def __init__(self,model_repo,prompt_set):
        model = CLIPModel.from_pretrained(model_repo)
        processor = CLIPProcessor.from_pretrained(model_repo)

    def wramup(self):
        pass

    def __call__(self,images):
        pass
