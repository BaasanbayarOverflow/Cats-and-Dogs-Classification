function setFileNameForInputLabel() {
    image = document.getElementById('file').files[0].name
    label = document.getElementById('file_label')
    label.innerHTML = image
}