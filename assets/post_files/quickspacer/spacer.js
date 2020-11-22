function getAllIndexes(arr, val) {
    var indexes = [], i;
    for (i = 0; i < arr.length; i++)
        if (arr[i] === val)
            indexes.push(i);
    return indexes;
}

window.onload = async function () {
    document.getElementById('original-text').value = [
        '누가나한테우유를던졌어아주신선한우유를말이야',
        '근데이거진짜로진짜제대로잘돌아가는거맞냐는게바로나의질문이란말이지',
        '',
        '아니근데이건좀너무한거아닙니까?',
        '맞춤법좀잘맞춰서띄어쓰기좀해여'
    ].join("\n")

    const vocabData = (await (await fetch('https://raw.githubusercontent.com/psj8252/quickspacer/master/quickspacer/resources/vocab.txt')).text()).split('\n');
    const vocabTable = Object.keys(vocabData).reduce((result, key) => { result[vocabData[key]] = parseInt(key); return result; }, {});
    const model1 = await tf.loadGraphModel('https://raw.githubusercontent.com/psj8252/quickspacer/master/tfjs_models/model1/model.json');
    const model2 = await tf.loadGraphModel('https://raw.githubusercontent.com/psj8252/quickspacer/master/tfjs_models/model2/model.json');
    const model3 = await tf.loadGraphModel('https://raw.githubusercontent.com/psj8252/quickspacer/master/tfjs_models/model3/model.json');

    model_inference = async function (model) {
        var startTime = Date.now()
        var inputText = document.getElementById('original-text').value.split("\n");

        inputArray = inputText.map(value => value.split(""))
        inputTensor = inputArray.map(t => t.map(key => vocabTable[key]));
        maxLength = Math.max(...inputTensor.map(t => t.length));
        inputTensor = inputTensor.map(t => [...t, ...Array(maxLength - t.length).fill(0)])
        inputTensor = tf.tensor2d(inputTensor, [inputTensor.length, maxLength], dtype = 'int32')

        output = await (model.predict(inputTensor))
        indicesToSpace = (await output.greater(0.5).array()).map(x => getAllIndexes(x, 1))

        result = inputArray.map((sentence, i) => {
            for (index of indicesToSpace[i]) {
                sentence[index] += " "
            }
            return sentence.join("")
        }).join("\n")

        const elaspsedTime = Date.now() - startTime;
        document.getElementById('elaspsed-time').innerHTML = "Elaspsed Time: " + elaspsedTime + " milliseconds";
        document.getElementById('spaced-text').value = result;
    }

    document.getElementById('remove-space-button').onclick = async function () {
        text = document.getElementById('original-text').value
        document.getElementById('original-text').value = text.split(" ").join("")
    }

    document.getElementById('submit-button1').onclick = () => model_inference(model1)
    document.getElementById('submit-button2').onclick = () => model_inference(model2)
    document.getElementById('submit-button3').onclick = () => model_inference(model3)
}
