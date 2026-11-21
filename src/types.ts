export interface ProjectFile {
  name: string;
  content: string;
  language: string;
}

export interface ProjectTemplate {
  id: string;
  title: string;
  description: string;
  category: string;
  icon: string;
  files: ProjectFile[];
  hyperparameters?: {
    epochs: number;
    batchSize: number;
    learningRate: number;
    optimizer: string;
    modelArchitecture: string;
  };
}
