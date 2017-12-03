

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
	$scope.hide_placeholder = true;


	$scope.samples_cardiomegaly = []
	$scope.samples_cardiomegaly.push('/static/samples/cardiomegaly/00000330_000.png')
	$scope.samples_cardiomegaly.push('/static/samples/cardiomegaly/00000211_038.png')
	$scope.samples_cardiomegaly.push('/static/samples/cardiomegaly/00000294_000.png')

	$scope.samples_pneumonia = []
	$scope.samples_pneumonia.push('/static/samples/pneumonia/00000061_015.png')
	$scope.samples_pneumonia.push('/static/samples/pneumonia/00001182_004.png')
	$scope.samples_pneumonia.push('/static/samples/pneumonia/00000165_001.png')

	$scope.samples_pneumothorax = []
	$scope.samples_pneumothorax.push('/static/samples/pneumothorax/00000296_000.png')
	$scope.samples_pneumothorax.push('/static/samples/pneumothorax/00000296_002.png')
	$scope.samples_pneumothorax.push('/static/samples/pneumothorax/00001006_020.png')

	$scope.result = {'colormap_path': null, 'class_probs':null};


	function get_class(index){

		if (index==0) {
			return "Pneumonia";
		}
		if (index==1) {
			return "Pneumothorax";
		}
		if (index==2) {
			return "Cardiomegaly";
		}
		if (index==3) {
			return "Pleural Thickening";
		}
		if (index==4) {
			return "Mass";
		}
	}

	$scope.select_pics = function(){
		$scope.hide_placeholder = false;
		$scope.show_sample = false;
		console.log('show pic selected');
	}

	$scope.uploadFiles = function(){
		$scope.hide_placeholder = false;
		console.log("sending file for prediction");
		$scope.show_sample = false;
		Upload.upload({

	        url: 'http://localhost:8000/demo/get_prediction/',
	        headers: { 'Content-Type': false,},
	        file: $scope.files

	    }).then(function(success_data){
	    	
	    	console.log(success_data);
	    	$scope.result = success_data.data;
	    	var probs = $scope.result.class_probs;
	    	$scope.result.class_probs = [];
	    	$scope.result.class_probs.push("Pneumonia: "+probs[0]);
	    	$scope.result.class_probs.push("Pneumothorax: "+probs[1]);
	    	$scope.result.class_probs.push("Cardiomegaly: "+probs[2]);
	    	$scope.result.class_probs.push("Pleural Thickening: "+probs[3]);
	    	$scope.result.class_probs.push("Mass: "+probs[4]);

	    	$scope.result.class = "Class Detected: "+get_class($scope.result.class[0]);
	    	

	    });
	};

    $scope.get_prediction = function (path) {

    	$scope.hide_placeholder = false;
    	dir_path = "./chexnet"+path
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
            
            $scope.result = data;
            var probs = $scope.result.class_probs;
            $scope.result.class_probs = [];
            $scope.result.class_probs.push("Pneumonia: "+probs[0]);
	    	$scope.result.class_probs.push("Pneumothorax: "+probs[1]);
	    	$scope.result.class_probs.push("Cardiomegaly: "+probs[2]);
	    	$scope.result.class_probs.push("Pleural Thickening: "+probs[3]);
	    	$scope.result.class_probs.push("Mass: "+probs[4]);

	    	$scope.result.class = "Class Detected: "+get_class($scope.result.class[0]);
            
        });
    };
	

}


