$(document).ready(function () {
    setTimeout(function () {
        $('#flashMessage').hide();
    }, 10000);

    $("#selectSlaTime").change(function() {
        let search = $(this).find("option:selected").attr("value");
        var data = 3000;
        let myArray = configData["RESOLUTION_SLA_TIME"];
       
        for (var i in myArray)
        {
            var id = myArray[i].id
            var value = myArray[i].value

            if (id == search)
            {
                data = value;
            }
           
        }
        console.log(data)
        $("#resolution_sla_time").val(
           


            data
           
            );
    });
  });