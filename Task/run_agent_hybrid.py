import click
import json
import dspy
import os
from dotenv import load_dotenv
from agent.graph_hybrid import HybridAgent

@click.command()
@click.option('--batch', required=True, help='Input JSONL file')
@click.option('--out', required=True, help='Output JSONL file')
def main(batch, out):
    """Run the retail analytics copilot."""
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    # Configure DSPy with OpenAI (correct class name)
    lm = dspy.LM(
        model='openai/gpt-4o-mini',  # or 'openai/gpt-4' for better quality
        api_key=api_key,
        max_tokens=500
    )
    dspy.settings.configure(lm=lm)
    
    # Initialize agent
    agent = HybridAgent(
        db_path='data/northwind.sqlite',
        docs_dir='docs'
    )
    
    # Process batch
    results = []
    with open(batch, 'r') as f:
        for line in f:
            question_data = json.loads(line)
            print(f"Processing: {question_data['id']}")
            
            result = agent.run(
                question=question_data['question'],
                format_hint=question_data['format_hint']
            )
            
            output = {
                "id": question_data['id'],
                **result
            }
            results.append(output)
    
    # Write outputs
    with open(out, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Results written to {out}")

if __name__ == '__main__':
    main()


# ============================================================================
# SETUP INSTRUCTIONS (OPENAI VERSION)
# ============================================================================
"""
1. Get OpenAI API key from https://platform.openai.com/api-keys
2. Create .env file with: OPENAI_API_KEY=your_key_here
3. Create project structure and files as shown above
4. Create docs files (marketing_calendar.md, kpi_definitions.md, etc.)
5. Download database:
   curl -L -o data/northwind.sqlite https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db
6. Create sample_questions_hybrid_eval.jsonl with the 6 questions from assignment
7. Install dependencies: pip install -r requirements.txt
8. Run: python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl

NOTES:
- Uses gpt-4o-mini by default (faster, cheaper)
- Can switch to gpt-4 for better accuracy: model='gpt-4'
- No local model installation needed
- Requires internet connection and OpenAI API credits
"""