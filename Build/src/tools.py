"""
Custom tool implementations for our agents.
"""
import datetime
import json
import os
import time
from typing import Dict, Any, List, Optional, Union, Literal

import replicate
import requests

from agents import function_tool, RunContextWrapper

@function_tool
def get_current_time(location: Optional[str]) -> str:
    """
    Get the current date and time.
    
    Args:
        location: Optional location name to get time for (currently ignored, always returns local time)
    
    Returns:
        A string representing the current date and time
    """
    # Force the system to use the current time
    # Get multiple time representations to debug any caching issues
    current_utc = datetime.datetime.utcnow()
    current_time = datetime.datetime.now()
    timestamp = time.time()
    
    # Format times in multiple ways to verify correctness
    formatted_time = current_time.strftime("%A, %B %d, %Y at %I:%M:%S %p")
    utc_time = current_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    iso_format = current_time.isoformat()
    
    # Create a comprehensive response with multiple time formats
    response = f"Current time: {formatted_time}\n"
    response += f"UTC time: {utc_time}\n"
    response += f"ISO format: {iso_format}\n"
    response += f"Unix timestamp: {timestamp}\n"
    response += f"Year: {current_time.year}, Month: {current_time.month}, Day: {current_time.day}"
    
    if location:
        return f"Current time for {location}:\n{response}"
    return response

@function_tool
def calculate_days_between(start_date: str, end_date: str) -> str:
    """
    Calculate the number of days between two dates.
    
    Args:
        start_date: Start date in format YYYY-MM-DD
        end_date: End date in format YYYY-MM-DD
    
    Returns:
        The number of days between the two dates
    """
    try:
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        
        difference = end - start
        return f"There are {difference.days} days between {start_date} and {end_date}"
    except ValueError as e:
        return f"Error parsing dates: {str(e)}. Please ensure dates are in format YYYY-MM-DD."
        
@function_tool
def get_todays_date() -> str:
    """
    Get today's date in multiple formats to verify system time.
    
    Returns:
        A string with today's date in multiple formats
    """
    now = datetime.datetime.now()
    
    # Use multiple date libraries and approaches to ensure we get the correct date
    system_date = now.strftime("%Y-%m-%d")
    formatted_date = now.strftime("%A, %B %d, %Y")
    timestamp = time.time()
    time_components = {
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "minute": now.minute
    }
    
    response = f"Today's date (YYYY-MM-DD): {system_date}\n"
    response += f"Today's date (formatted): {formatted_date}\n"
    response += f"Current timestamp: {timestamp}\n"
    response += f"Date components: {time_components}"
    
    return response

@function_tool
def format_data(ctx: RunContextWrapper[Any], data_str: str, format_type: str) -> str:
    """
    Format data into various formats.
    
    Args:
        data_str: JSON string representation of data to format
        format_type: The format to convert to (json or text)
    
    Returns:
        Formatted data as a string
    """
    try:
        # Parse the JSON string into a Python dictionary
        data = json.loads(data_str)
        
        if format_type.lower() == "json":
            return json.dumps(data, indent=2)
        elif format_type.lower() == "text":
            # Convert dictionary to text format
            return "\n".join([f"{key}: {value}" for key, value in data.items()])
        else:
            return f"Unsupported format: {format_type}. Please use 'json' or 'text'."
    except json.JSONDecodeError:
        return f"Error: Invalid JSON string provided. Please provide a valid JSON string."
        
@function_tool
async def generate_image(
    prompt: str, 
    aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4"] = "1:1"
) -> str:
    """
    Generate an image using the Replicate Flux-1.1-Pro model.
    
    Args:
        prompt: A detailed description of the image you want to generate
        aspect_ratio: The aspect ratio of the output image - options are "1:1" (square), "16:9" (landscape), 
                      "9:16" (portrait), "4:3" (landscape), "3:4" (portrait).
                      
    Returns:
        A message with the image URL once generated
    """
    # Check if API token is available
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        return "ERROR: REPLICATE_API_TOKEN environment variable is not set."
    
    try:
        # Start the prediction
        prediction = replicate.predictions.create(
            model="black-forest-labs/flux-1.1-pro",
            input={
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": "png",
                "output_quality": 80,
                "safety_tolerance": 2,
                "prompt_upsampling": True
            }
        )
        
        # Get the prediction ID
        prediction_id = prediction.id
        
        # Wait for the prediction to complete (poll every 2 seconds)
        max_attempts = 15  # Maximum number of attempts (30 seconds total)
        attempts = 0
        
        while attempts < max_attempts:
            # Get the prediction status
            prediction = replicate.predictions.get(prediction_id)
            
            # Check if the prediction is complete
            if prediction.status == "succeeded":
                # Return the image URL
                return f"Image generated successfully! Here's your image: {prediction.output}"
            
            # Check if the prediction failed
            elif prediction.status == "failed":
                return f"Image generation failed: {prediction.error}"
            
            # Wait before checking again
            time.sleep(2)
            attempts += 1
        
        return f"Image generation is taking longer than expected. You can check the status later with this ID: {prediction_id}"
    
    except Exception as e:
        return f"Error generating image: {str(e)}"