//function showMenu() {
//  var x = document.getElementById("left").classList.remove('d-none');
//  if (x.style.display === "none") {
//    x.style.display = "block";
//  } else {
//    x.style.display = "none";
//  }
//}

function ShowHideMenu() {
       var x = document.getElementById('left').classList.remove('d-none');
       if(x.style.display == 'none')
          x.style.display = 'block';
       else
          x.style.display = 'none';
    }