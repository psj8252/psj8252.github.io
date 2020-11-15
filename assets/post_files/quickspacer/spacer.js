function getAllIndexes(arr, val) {
    var indexes = [], i;
    for (i = 0; i < arr.length; i++)
        if (arr[i] === val)
            indexes.push(i);
    return indexes;
}

window.onload = async function () {
    document.getElementById('original-text').value = [
        '마춤법좀잘마쳐서띄어쓰기좀해여',
        '',
        '아니근데이건좀너무한거아닙니까?',
    ].join("\n")

    const model = await tf.loadGraphModel('../assets/post_files/quickspacer/model.json');
    const vocabData = (await (await fetch('../assets/post_files/quickspacer/vocab.txt')).text()).split('\n');
    const vocabTable = Object.keys(vocabData).reduce((result, key) => { result[vocabData[key]] = parseInt(key); return result; }, {});

    document.getElementById('submit-button').onclick = async function () {
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

        document.getElementById('spaced-text').value = result
    }
}
