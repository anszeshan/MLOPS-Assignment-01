pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE_NAME = 'assignment01:frontend'
    }

    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    sh "docker build -t $flask-app ."
                }
            }
        }

        stage('Login Dockerhub and Push Docker Image') {
            environment {
                DOCKER_HUB_CREDENTIALS = credentials('dockerhub-credentials')
            }
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials', usernameVariable: 'DOCKER_HUB_USERNAME', passwordVariable: 'DOCKER_HUB_PASSWORD')]) {
                        sh "echo $DOCKER_HUB_PASSWORD | docker login -u $DOCKER_HUB_USERNAME --password-stdin"
                        
                        sh "docker push $flask-app"
                    }
                }
            }
        }
    }
}
