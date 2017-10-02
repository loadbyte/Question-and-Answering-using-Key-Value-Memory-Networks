$(function() {
    var $story        = $('#story'),
        $question     = $('#question'),
        $answer       = $('#answer'),
        $getAnswer    = $('#get_answer'),
        $getStory     = $('#get_story'),
        $explainTable = $('#explanation');

    getStory();

    
    $('.qa-container').find('.glyphicon-info-sign').tooltip();

    $getAnswer.on('click', function(e) {
        e.preventDefault();
        getAnswer();
    });

    $getStory.on('click', function(e) {
        e.preventDefault();
        getStory();
    });
    // On click get Story from the server.ss
    function getStory() {
        $.get('/get/story', function(json) {
            $story.val(json["story"]);
            $question.val(json["question"]);
            $question.data('question_idx', json["question_idx"]);
            $answer.val('');
            
        });
    }

    // On click get the predicted output from server.
    function getAnswer() {
        var question = $question.val();
        var predAnswer='';
        $answer.val('');
        var questionIdx = $question.data('question_idx');
        var url = '/get/answer?question_idx=' + questionIdx +'&user_question=' + encodeURIComponent(question);
        $.get(url, function(json) {
            predAnswer = json["pred_answer"];

            var outputMessage = "Answer = '" + predAnswer + "'";           
            $answer.val(outputMessage);

            
        });
        
    }
});
