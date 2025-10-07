# debug_embed.py
import cohere
from app.config import settings

co = cohere.ClientV2(api_key=settings.COHERE_API_KEY)

def debug_embed():
    import inspect
    print("Signature of co.embed:", inspect.signature(co.embed))
    # Help text
    print("Doc:", co.embed.__doc__)

if __name__ == "__main__":
    debug_embed()
