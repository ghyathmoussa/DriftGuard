#!/usr/bin/env python3
"""
DriftGuard Demo Launcher
Interactive menu to run different demos
"""

import subprocess
import sys

def print_header():
    print("\n" + "=" * 70)
    print("🛡️  DRIFTGUARD - ML Model Drift Detection & Monitoring")
    print("=" * 70)

def print_menu():
    print("\n📋 Available Demos:\n")
    print("  1. Simple LLM Drift Detection (Quick Start)")
    print("     → Basic drift detection with examples")
    print()
    print("  2. Comprehensive Feature Demo")
    print("     → Data drift, concept drift, performance monitoring")
    print()
    print("  3. Real-World Chatbot Scenario")
    print("     → Week-by-week monitoring simulation")
    print()
    print("  4. Original LLM Demo")
    print("     → LLM metrics and drift detection")
    print()
    print("  5. Traditional ML Monitoring")
    print("     → Performance tracking demo")
    print()
    print("  6. Concept Drift Visualization")
    print("     → ADWIN drift detection with plots")
    print()
    print("  7. Run All Demos (Sequential)")
    print()
    print("  0. Exit")
    print()

def run_demo(script_path):
    """Run a demo script"""
    try:
        print(f"\n🚀 Running: {script_path}")
        print("-" * 70)
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            print("-" * 70)
            print("✅ Demo completed successfully!")
        else:
            print("-" * 70)
            print("⚠️  Demo finished with warnings or errors")
        
        input("\nPress Enter to continue...")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        input("\nPress Enter to return to menu...")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        input("\nPress Enter to continue...")

def run_all_demos():
    """Run all demos sequentially"""
    demos = [
        "demo_simple.py",
        "demo_comprehensive.py",
        "demo_chatbot.py",
        "examples/llm_demo.py"
    ]
    
    print("\n🚀 Running all demos sequentially...")
    print("=" * 70)
    
    for i, demo in enumerate(demos, 1):
        print(f"\n[{i}/{len(demos)}] Starting: {demo}")
        run_demo(demo)
    
    print("\n" + "=" * 70)
    print("✅ All demos completed!")
    print("=" * 70)

def main():
    """Main menu loop"""
    while True:
        print_header()
        print_menu()
        
        try:
            choice = input("Select demo (0-7): ").strip()
            
            if choice == "0":
                print("\n👋 Goodbye! Happy monitoring!")
                break
            
            elif choice == "1":
                run_demo("demo_simple.py")
            
            elif choice == "2":
                run_demo("demo_comprehensive.py")
            
            elif choice == "3":
                run_demo("demo_chatbot.py")
            
            elif choice == "4":
                run_demo("examples/llm_demo.py")
            
            elif choice == "5":
                run_demo("examples/monitoring.py")
            
            elif choice == "6":
                run_demo("examples/concept_drift.py")
            
            elif choice == "7":
                run_all_demos()
            
            else:
                print("\n⚠️  Invalid choice. Please select 0-7.")
                input("Press Enter to continue...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Happy monitoring!")
            break
        except EOFError:
            break

if __name__ == "__main__":
    main()

