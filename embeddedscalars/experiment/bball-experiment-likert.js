var sentences = [
'No player hit none of his shots',
'No player hit some of his shots',
'No player hit all of his shots',
'Exactly one player hit none of his shots',
'Exactly one player hit some of his shots',
'Exactly one player hit all of his shots',
'Every player hit none of his shots',
'Every player hit some of his shots',
'Every player hit all of his shots'];

var fillers = [
'All the players are tied',
'Player A shot perfectly',
'Player B shot perfectly',
'Player C shot perfectly',
'Player A made no shots',
'Player B made no shots',
'Player C made no shots',
'We have a clear winner',
'We have a clear loser',
'Player A tied with Player B',
'Player A tied with Player C',
'Player B tied with Player C',
'Most of the players made shots',
'Most of the players missed shots',
'Someone did better than Player A',
'Someone did better than Player B',
'Someone did better than Player C',
'Player A placed second',
'Player B placed second',
'Player C placed second'];

//var conditions = expand(['none','few','half','most','all']);
var conditions = [
['none','none','none'],
['none','none','half'],
['none','none','all'],
['none','half','half'],
['none','half','all'],
['none','all','all'],
['half','half','half'],
['half','half','all'],
['half','all','all'],
['all','all','all']
];

var colors = ['brown','darkorange','grey','lavender','lightblue','lightorange','midnight','pink','purple','royalblue','seafoam','tan','wheat','yellow']; 

var myStimuli = shuffle(sentences.concat(fillers));

var numberOfQuestions = myStimuli.length;

var counter = -3; // 3 training trials, then main event.

$(document).ready(function() {
    showSlide("instructions");
    $("#instructions #mustaccept").hide();
});

var experiment = {
    data: [],
    intro: function () {
    	if (turk.previewMode) {
	    	$("#instructions #mustaccept").show();
		} else {
	    	showSlide("intro");
	    	$("#introbutton").click(function(){
    			showSlide('training0'); 
    			setTimeout(function(){
    				experiment.next();
    			}, 2000);	
    		});
	    }
    },
    next: function() {
    	if (myStimuli.length == 0) { 
		    showSlide("demographic");
		    $("#lgerror").hide();
		    $("#lgbox").keypress(function(e){ 
// capture return so that it doesn't restart experiment
		    	if (e.which == 13) {
		    		return false;
		    	}
		    });
		    $("#lgsubmit").click(function(){
				var lang = document.getElementById("lgbox").value;
				if (lang.length > 5) {
				    showSlide("finished");
				    setTimeout(function() {turk.submit({
				    	data: experiment.data,
				    	language: lang
				    })}, 1000);
				}
				return false;
			});
		} else { 
			$('#feedback').hide();
			$('#error').hide();
			$('.rating').attr('checked',false);
			if (counter < 0) {
				$('.progress').hide();
				$('#toptext').show();
			} else {
				$('#toptext').hide();
				$('.progress').show();
			}
			qdata = {};
			var myColors = shuffle(colors);
			qdata.player1color = myColors.shift();
			qdata.player2color = myColors.shift();
			qdata.player3color = myColors.shift();
			if (counter >= 0) {
				var condition = conditions.random();
		    	qdata.sentence = myStimuli.shift();
		    	var conditionShuffled = shuffle(condition);
		    } else if (counter == -3) {
		    	$('#toptext').html('<b>First practice question:</b>');
		    	var condition = ['all','none','few'];
		    	qdata.sentence = "Player A made none of his baskets";
		    	var conditionShuffled = condition;
		    } else if (counter == -2) {
		    	$('#toptext').html('<b>Second practice question:</b>');
		    	var condition = ['all','all','none'];
		    	qdata.sentence = "Player C messed up";
		    	var conditionShuffled = condition;
		    } else {
		    	$('#toptext').html('<b>Last practice question:</b>');
		    	var condition = ['most','none','all'];
		    	qdata.sentence = "Player A was the star player of the round";
		    	var conditionShuffled = condition;
		    }
		    qdata.condition = condition[0] + '-' + condition[1] + '-' + condition[2];
		    qdata.conditionOrder = conditionShuffled[0] + '-' + conditionShuffled[1] + '-' + conditionShuffled[2];
		    var player1baskets = conditionShuffled[0];
		    var player2baskets = conditionShuffled[1];
		    var player3baskets = conditionShuffled[2];
	    	$("#images").html('<td><img class="player imgs1" src="basketball_images/player_' + qdata.player1color + '.png"><img class="balls imgs1" src="basketball_images/balls_' + player1baskets + '.png"></td><td><img class="player imgs2" src="basketball_images/player_' + qdata.player2color + '.png"><img class="balls imgs2" src="basketball_images/balls_' + player2baskets + '.png"></td><td><img class="player imgs3" src="basketball_images/player_' + qdata.player3color + '.png"><img class="balls imgs3" src="basketball_images/balls_' + player3baskets + '.png"></td>');
		    $("#questiontxt").html(qdata.sentence + '.');
	    	showSlide("stage");
	    	var startTime = (new Date()).getTime();  
	    	qdata.trialnum = counter;
	    	var trainingTrialCorrect = -1;
	    	$("#continue").click(function() {
	    		var response = $(".rating").serialize();
	    		if (response.length < 8) { 
	    			$('#error').show();
	    		} else { 
	    			$('#error').hide();
	    			if (counter < 0) { // if we're in a training trial
	    				if (isCorrect(response[7],counter)) {
	    					if (trainingTrialCorrect != "no") {
	    						trainingTrialCorrect = "yes";
	    					} 
	    					$('#feedback').html("Good answer!");
	    					$('#feedback').show();
	    					$("#continue").unbind('click');
	    					setTimeout(function(){
	    						counter = counter + 1;
	    						qdata.response = response[7];
	    						var endTime = (new Date()).getTime();
	    						qdata.rt = endTime - startTime;
	    						qdata.trainingCorrect = trainingTrialCorrect;
	    						experiment.data.push(qdata);
	    						if (counter == 0) {
	    							showSlide("training4");
	    							setTimeout(function(){
	    								showSlide("stage");
	    								experiment.next();
	    							}, 3000);
	    						} else {
	    						    experiment.next()
	    						}
	    					}, 2000);
	    				} else {
	    					trainingTrialCorrect = "no";
	    					$('#feedback').html("Hmm, that doesn't seem right. Can you try again?");
	    					$('#feedback').show();
	    				}
	    			} else { // advance to next question
	    				$("#continue").unbind('click');
	    				counter = counter + 1;
	    				qdata.response = response[7];
	    				var endTime = (new Date()).getTime();
	    				qdata.rt = endTime - startTime;
	    				qdata.trainingCorrect = "NA";
	    				$('.bar').css('width', (200.0 * counter / numberOfQuestions) + 'px');
	    				experiment.data.push(qdata); 
	    // add trial data to experiment.data object, which is what we'll eventually submit to MTurk
	    				experiment.next();
	    			}
	    		}
	    	});   	
	    }
	}
}

function isCorrect(response,num) {
	if (num == -3 && response < 3) {return true;} 
// first training trial is clearly false
	else if (num == -2 && response > 5) {return true;}
// second training trial is clearly true
	else if (num == -1 && response < 3) {return true;}
// third training trial is clearly false
	else {return false;}
}

function showSlide(id) {
	$(".slide").hide();
	$("#"+id).show();
	return false;
};

function shuffle(v) { // non-destructive.
    newarray = v.slice(0);
    for(var j, x, i = newarray.length; i; j = parseInt(Math.random() * i), x = newarray[--i], newarray[i] = newarray[j], newarray[j] = x);
    return newarray;
};
function random(a,b) {
    if (typeof b == "undefined") {
		a = a || 2;
		return Math.floor(Math.random()*a);
    } else {
		return Math.floor(Math.random()*(b-a+1)) + a;
    }
};
Array.prototype.random = function() { return this[random(this.length)]; }