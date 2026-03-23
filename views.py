import os
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .script import predictrf, predictlr, predictsvr, pm25_to_aqi, aqi_level

# หน้าเว็บหลัก
def pm_view(request):
    if request.method == "POST" and request.FILES.get("imagepath"):
        try:
            imagepath = request.FILES["imagepath"]

            upload_dir = os.path.join(settings.BASE_DIR, "imagestorage")
            os.makedirs(upload_dir, exist_ok=True)

            fs = FileSystemStorage(location=upload_dir)
            filename = fs.save(imagepath.name, imagepath)
            newfilepath = fs.path(filename)

            xrf = predictrf(newfilepath)
            xlr = predictlr(newfilepath)
            xsvr = predictsvr(newfilepath)

            aqi_rf = pm25_to_aqi(xrf)
            aqi_lr = pm25_to_aqi(xlr)
            aqi_svr = pm25_to_aqi(xsvr)
            aqi_avg = round((aqi_rf + aqi_lr + aqi_svr) / 3)
            level_text, level_color = aqi_level(aqi_avg)

            file_url = "/imagestorage/" + filename

            # 👇 เก็บ context ลง session
            request.session["context"] = {
                "pm_rf": xrf,
                "pm_lr": xlr,
                "pm_svr": xsvr,
                "aqi_rf": aqi_rf,
                "aqi_lr": aqi_lr,
                "aqi_svr": aqi_svr,
                "aqi_avg": aqi_avg,
                "aqi_level": level_text,
                "aqi_color": level_color,
                "filename": filename,
                "file_url": file_url,
            }

            return redirect("pm_view") 

        except (TypeError, ValueError):
            return redirect("pm_view")

    context = request.session.pop("context", {
        "pm_rf": None,
        "pm_lr": None,
        "pm_svr": None,
        "aqi_rf": None,
        "aqi_lr": None,
        "aqi_svr": None,
        "aqi_avg": None,
        "aqi_level": None,
        "aqi_color": None,
        "filename": None,
        "file_url": None,
    })

    return render(request, "pm.html", context)
