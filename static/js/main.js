$(document).ready(function (){

    $nav = $('.nav');
    $toggleCollapse = $('.toggle-collapse');

   
    $toggleCollapse.click(function (){
        $nav.toggleClass('collapse');
    })


$('.owl-carousel').owlCarousel({
    loop: true,
    autoplay: true,
    autoplayTimeout: 3000,
    dots: false,
    nav: true,
    navText: [$('.owl-navigation .owl-nav-prev'), $('.owl-navigation .owl-nav-next')]
});


});