from django.shortcuts import render
from . import ml2

# Create your views here.
def index(request):    
    if request.method == "POST":
        new_text = ml2.detector1(source=request.POST.get("text1", None), title=request.POST.get("text2", None), text=request.POST.get("text3", None))
        pass
    else:
        return render(request, 'fakeNews/index.html', {'result1': '', 'result2': '', 'result3': '', 'accuracy': ml2.get_accuracy()})
    return render(request, 'fakeNews/index.html', {'result1': new_text[0], 'result2': new_text[1], 'result3': new_text[2], 'accuracy': ml2.get_accuracy()})
    

