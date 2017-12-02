

var mainApp = angular.module('mainApp', ['ngMaterial', 'ngFileUpload']);


mainApp.config(function($locationProvider) {
    $locationProvider.html5Mode(true);
});
 
mainApp.config(function($interpolateProvider){
    $interpolateProvider.startSymbol('[[').endSymbol(']]');
});


mainApp.controller('homePageController', homePageController);
function homePageController($scope, $rootScope, $timeout, Upload, $http){

	
	$scope.show_sample = false;
	$scope.sample_image = null;


	$scope.samples_cardiomegaly = []
	$scope.samples_cardiomegaly.push('/static/samples/cardiomegaly/00000001_000.png')
	$scope.samples_cardiomegaly.push('/static/samples/cardiomegaly/00000233_000.png')
	$scope.samples_cardiomegaly.push('/static/samples/pneumothorax/00000416_005.png')


	$scope.result = {'colormap_path': null, 'class_probs':null};

	$scope.select_pics = function(){
		$scope.show_sample = false;
		console.log('show pic selected');
	}

	$scope.uploadFiles = function(){

		Upload.upload({

	        url: 'http://localhost:8000/demo/get_prediction/',
	        headers: { 'Content-Type': false,},
	        file: $scope.files

	    }).then(function(success_data){
	    	
	    	console.log(success_data);
	    	$scope.result = success_data.data;


	    });
	};

    $scope.get_prediction = function (path) {

    	dir_path = "./chexnet/"+path
    	$scope.sample_image = path;
    	$scope.show_sample = true;
    	$scope.sample_image = path;

    	print (dir_path)
       // use $.param jQuery function to serialize data from JSON 
        var data = $.param({
            image_path: dir_path
        });
    
        var config = {
            headers : {
                'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8;'
            }
        }

        $http.post('http://localhost:8000/demo/get_prediction/', data, config)
        .success(function (data, status, headers, config) {
            console.log(data);
            $scope.result = data;
            
        });
    };
	

}


