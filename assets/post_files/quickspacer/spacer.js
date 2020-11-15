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

    const model = await tf.loadGraphModel('../assets/post_files/quickspacer/model.json');
    const vocabData = (await (await fetch('../assets/post_files/quickspacer/vocab.txt')).text()).split('\n');
    const vocabTable = Object.keys(vocabData).reduce((result, key) => { result[vocabData[key]] = parseInt(key); return result; }, {});

    document.getElementById('submit-button').onclick = async function () {
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
}
