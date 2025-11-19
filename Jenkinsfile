pipeline {
    // Later we can change this to a GPU node label if you add agents
    agent any

    environment {
        PYTHON    = "python3"
        VENV_DIR  = ".venv"
        OUTPUT_DIR = "artifacts"
    }

    stages {
        stage('Checkout') {
            steps {
                // when using "Pipeline from SCM", Jenkins already checks out,
                // but this doesn't hurt and keeps it clear.
                checkout scm
            }
        }

        stage('Setup Python Env') {
            steps {
                sh """
                ${PYTHON} -m venv ${VENV_DIR}
                . ${VENV_DIR}/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                """
            }
        }

        stage('Train Model') {
            steps {
                sh """
                . ${VENV_DIR}/bin/activate
                ${PYTHON} train.py --epochs 2 --output_dir ${OUTPUT_DIR}
                """
            }
        }

        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: "${OUTPUT_DIR}/**", fingerprint: true
            }
        }
    }

    post {
        success {
            echo " Training completed successfully."
        }
        failure {
            echo " Training failed. Check console output."
        }
    }
}
