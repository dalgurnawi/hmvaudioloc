from src.models.model_architectures.model_baseline_vanilla import create_baseline_vanilla_model
from src.models.model_architectures.model_baseline import create_baseline_model
from src.models.model_architectures.model_baseline_gaussian_noise import create_baseline_gaussian_noise_model

from src.models.model_architectures.model_paper_code_1 import create_paper_code_1_model
from src.models.model_architectures.model_paper_code_2 import create_paper_code_2_model
from src.models.model_architectures.model_paper_code_3 import create_paper_code_3_model
from src.models.model_architectures.model_paper_code_4 import create_paper_code_4_model
from src.models.model_architectures.model_paper_code_5 import create_paper_code_5_model
from src.models.model_architectures.model_paper_code_6 import create_paper_code_6_model
from src.models.model_architectures.model_paper_code_7 import create_paper_code_7_model
from src.models.model_architectures.model_paper_code_8 import create_paper_code_8_model
from src.models.model_architectures.model_paper_code_9 import create_paper_code_9_model
from src.models.model_architectures.model_paper_code_10 import create_paper_code_10_model
from src.models.model_architectures.model_paper_code_11 import create_paper_code_11_model

model = create_baseline_model()
model.summary()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# history = model.fit(
#     dataset_train,
#     epochs=epochs,
#     validation_data=dataset_validate
# )
