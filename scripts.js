var menuIcon = document.querySelector('.menu-icon');
var nav = document.querySelector('nav ul');

menuIcon.addEventListener('click', function() {
  menuIcon.classList.toggle('active');
  nav.classList.toggle('active');
});
